"""
Auto-documentation generator for COMPASS modules.

Scans ``compass/*.py`` via the AST (no imports, no side effects), extracts
class hierarchies, public methods, docstrings, inter-module dependencies,
and auto-detects test coverage from the ``tests/`` directory.

Usage::

    from compass.auto_docs import AutoDocGenerator
    gen = AutoDocGenerator()
    gen.scan()
    gen.generate("reports/compass_docs.html")
"""

from __future__ import annotations

import ast
import html as html_mod
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
COMPASS_DIR = ROOT / "compass"
TESTS_DIR = ROOT / "tests"
DEFAULT_OUTPUT = ROOT / "reports" / "compass_docs.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ParamInfo:
    """A single function/method parameter."""
    name: str
    annotation: str = ""
    default: str = ""


@dataclass
class MethodInfo:
    """A public method or module-level function."""
    name: str
    docstring: str = ""
    params: List[ParamInfo] = field(default_factory=list)
    returns: str = ""
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False


@dataclass
class ClassInfo:
    """A class extracted from a module."""
    name: str
    docstring: str = ""
    bases: List[str] = field(default_factory=list)
    init_params: List[ParamInfo] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Full metadata for one compass module."""
    filename: str
    name: str
    docstring: str = ""
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[MethodInfo] = field(default_factory=list)
    imports_from_compass: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    line_count: int = 0
    has_tests: bool = False
    test_files: List[str] = field(default_factory=list)
    test_count: int = 0


@dataclass
class CoverageInfo:
    """Test coverage summary for a module."""
    module_name: str
    has_tests: bool
    test_files: List[str]
    test_count: int


# ── AST helpers ──────────────────────────────────────────────────────────


def _unparse(node) -> str:
    """Best-effort unparse of an AST node."""
    if node is None:
        return ""
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return ""
    return ""


def _extract_params(args: ast.arguments) -> List[ParamInfo]:
    params: List[ParamInfo] = []
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, arg in enumerate(args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        ann = _unparse(arg.annotation)
        default_idx = i - (n_args - n_defaults)
        dflt = _unparse(args.defaults[default_idx]) if default_idx >= 0 else ""
        params.append(ParamInfo(arg.arg, ann, dflt))
    for arg, dflt in zip(args.kwonlyargs, args.kw_defaults):
        params.append(ParamInfo(arg.arg, _unparse(arg.annotation), _unparse(dflt) if dflt else ""))
    return params


def _extract_method(node: ast.FunctionDef) -> MethodInfo:
    decs = [_unparse(d) for d in node.decorator_list]
    return MethodInfo(
        name=node.name,
        docstring=ast.get_docstring(node) or "",
        params=_extract_params(node.args),
        returns=_unparse(node.returns),
        is_static="staticmethod" in decs,
        is_classmethod="classmethod" in decs,
        is_property="property" in decs,
    )


def _extract_class(node: ast.ClassDef) -> ClassInfo:
    bases = [_unparse(b) for b in node.bases]
    methods: List[MethodInfo] = []
    init_params: List[ParamInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                init_params = _extract_params(item.args)
            elif not item.name.startswith("_"):
                methods.append(_extract_method(item))
    return ClassInfo(
        name=node.name,
        docstring=ast.get_docstring(node) or "",
        bases=bases,
        init_params=init_params,
        methods=methods,
    )


def _extract_compass_imports(tree: ast.Module) -> List[str]:
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("compass."):
                imports.add(node.module.split(".")[1])
    return sorted(imports)


def _extract_constants(tree: ast.Module) -> List[str]:
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_"):
                    out.append(t.id)
    return out


# ── Module parsing ───────────────────────────────────────────────────────


def parse_module(filepath: Path) -> ModuleInfo:
    """Parse a single .py file into ModuleInfo via AST."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ModuleInfo(filename=filepath.name, name=filepath.stem, line_count=lines)

    classes = [_extract_class(n) for n in tree.body if isinstance(n, ast.ClassDef)]
    functions = [_extract_method(n) for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and not n.name.startswith("_")]

    return ModuleInfo(
        filename=filepath.name,
        name=filepath.stem,
        docstring=ast.get_docstring(tree) or "",
        classes=classes,
        functions=functions,
        imports_from_compass=_extract_compass_imports(tree),
        constants=_extract_constants(tree),
        line_count=lines,
    )


# ── Test coverage detection ──────────────────────────────────────────────


def detect_test_coverage(
    module_name: str,
    tests_dir: Path = TESTS_DIR,
) -> CoverageInfo:
    """Detect which test files cover a given compass module.

    Heuristics:
      1. Direct name match: test_{module_name}.py
      2. Grep test files for 'from compass.{module_name}' or 'import {module_name}'
    """
    test_files: List[str] = []
    test_count = 0

    if not tests_dir.exists():
        return CoverageInfo(module_name, False, [], 0)

    # 1. Direct match
    direct = tests_dir / f"test_{module_name}.py"
    if direct.exists():
        test_files.append(direct.name)
        test_count += _count_test_functions(direct)

    # 2. Pattern match: test files that import this module
    pattern = re.compile(rf"from\s+compass\.{re.escape(module_name)}\s+import|"
                         rf"import\s+compass\.{re.escape(module_name)}")
    for fp in tests_dir.glob("test_*.py"):
        if fp.name in [f"test_{module_name}.py"]:
            continue  # already counted
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            if pattern.search(text):
                test_files.append(fp.name)
                test_count += _count_test_functions(fp)
        except Exception:
            pass

    return CoverageInfo(
        module_name=module_name,
        has_tests=len(test_files) > 0,
        test_files=test_files,
        test_count=test_count,
    )


def _count_test_functions(filepath: Path) -> int:
    """Count test functions (def test_*) in a file."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        return len(re.findall(r"^\s*def\s+test_", text, re.MULTILINE))
    except Exception:
        return 0


# ── Dependency graph ─────────────────────────────────────────────────────


def build_dependency_graph(modules: List[ModuleInfo]) -> Dict[str, List[str]]:
    """Build {module: [modules it imports from]} dependency dict."""
    return {m.name: m.imports_from_compass for m in modules}


def render_text_dependency_graph(dep_graph: Dict[str, List[str]]) -> str:
    """Render a text-based dependency graph."""
    lines: List[str] = []
    for mod in sorted(dep_graph):
        deps = dep_graph[mod]
        if deps:
            lines.append(f"  {mod} → {', '.join(deps)}")
        else:
            lines.append(f"  {mod} (no compass deps)")
    return "\n".join(lines)


# ── AutoDocGenerator ─────────────────────────────────────────────────────


class AutoDocGenerator:
    """Scans COMPASS modules and generates documentation.

    Args:
        compass_dir: Path to the compass/ directory.
        tests_dir: Path to the tests/ directory.
    """

    def __init__(
        self,
        compass_dir: str = str(COMPASS_DIR),
        tests_dir: str = str(TESTS_DIR),
    ):
        self.compass_dir = Path(compass_dir)
        self.tests_dir = Path(tests_dir)
        self.modules: List[ModuleInfo] = []
        self.dep_graph: Dict[str, List[str]] = {}

    def scan(self) -> List[ModuleInfo]:
        """Scan all compass/*.py modules and detect test coverage."""
        self.modules = []
        for fp in sorted(self.compass_dir.glob("*.py")):
            if fp.name.startswith("__"):
                continue
            mod = parse_module(fp)
            cov = detect_test_coverage(mod.name, self.tests_dir)
            mod.has_tests = cov.has_tests
            mod.test_files = cov.test_files
            mod.test_count = cov.test_count
            self.modules.append(mod)

        self.dep_graph = build_dependency_graph(self.modules)
        logger.info("Scanned %d modules, %d with tests",
                     len(self.modules), sum(1 for m in self.modules if m.has_tests))
        return self.modules

    def coverage_summary(self) -> Dict[str, Any]:
        """Aggregate test coverage statistics."""
        n = len(self.modules)
        tested = sum(1 for m in self.modules if m.has_tests)
        total_tests = sum(m.test_count for m in self.modules)
        untested = [m.name for m in self.modules if not m.has_tests]
        return {
            "total_modules": n,
            "tested_modules": tested,
            "untested_modules": n - tested,
            "coverage_pct": round(tested / n * 100, 1) if n > 0 else 0,
            "total_tests": total_tests,
            "untested_names": untested,
        }

    def generate(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML documentation. Returns output path."""
        if not self.modules:
            self.scan()
        html = self._build_html()
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Docs written to %s (%d bytes)", out, len(html))
        return str(out.resolve())

    # ── HTML builder ─────────────────────────────────────────────────

    def _build_html(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        esc = html_mod.escape
        n_classes = sum(len(m.classes) for m in self.modules)
        n_funcs = sum(len(m.functions) for m in self.modules)
        cov = self.coverage_summary()

        # Module index
        index_rows = ""
        for m in self.modules:
            badge = '<span class="cov-yes">✓</span>' if m.has_tests else '<span class="cov-no">✗</span>'
            index_rows += (
                f'<tr><td><a href="#{esc(m.name)}">{esc(m.name)}</a></td>'
                f'<td>{len(m.classes)}</td><td>{len(m.functions)}</td>'
                f'<td>{m.line_count}</td><td>{badge} {m.test_count}</td></tr>\n'
            )

        # Module detail sections
        sections = ""
        for m in self.modules:
            doc_line = esc(m.docstring.split("\n")[0]) if m.docstring else ""
            deps_html = ", ".join(f'<a href="#{d}">{d}</a>' for d in m.imports_from_compass) or "<em>none</em>"
            tests_html = ", ".join(f'<code>{esc(t)}</code>' for t in m.test_files) or "<em>no tests</em>"
            consts = ", ".join(f'<code>{esc(c)}</code>' for c in m.constants[:8]) if m.constants else ""

            class_html = ""
            for cls in m.classes:
                init_sig = self._format_params(cls.init_params)
                bases = f' ({", ".join(esc(b) for b in cls.bases)})' if cls.bases else ""
                method_rows = ""
                for meth in cls.methods:
                    tag = ""
                    if meth.is_static:
                        tag = '<span class="tag">static</span> '
                    elif meth.is_classmethod:
                        tag = '<span class="tag">cls</span> '
                    elif meth.is_property:
                        tag = '<span class="tag">prop</span> '
                    sig = self._format_sig(meth)
                    doc = esc(meth.docstring.split("\n")[0]) if meth.docstring else ""
                    method_rows += f'<tr><td>{tag}{sig}</td><td class="doc-cell">{doc}</td></tr>\n'
                cls_doc = esc(cls.docstring.split("\n")[0]) if cls.docstring else ""
                class_html += f"""
                <div class="class-card">
                  <h4>class {esc(cls.name)}{bases}</h4>
                  <p class="doc">{cls_doc}</p>
                  {f'<p class="sig"><code>{esc(cls.name)}({init_sig})</code></p>' if init_sig else ''}
                  {'<table class="mtable"><thead><tr><th>Method</th><th>Description</th></tr></thead><tbody>' + method_rows + '</tbody></table>' if method_rows else ''}
                </div>"""

            func_html = ""
            for func in m.functions:
                sig = self._format_sig(func)
                doc = esc(func.docstring.split("\n")[0]) if func.docstring else ""
                func_html += f'<div class="func-item"><code>{sig}</code><span class="doc"> — {doc}</span></div>\n'

            sections += f"""
            <div class="module-section" id="{esc(m.name)}">
              <h3>{esc(m.filename)} <small>({m.line_count} lines)</small></h3>
              <p class="doc">{doc_line}</p>
              <p class="meta-line"><strong>Depends on:</strong> {deps_html}</p>
              <p class="meta-line"><strong>Tests:</strong> {tests_html} ({m.test_count} tests)</p>
              {f'<p class="meta-line"><strong>Constants:</strong> {consts}</p>' if consts else ''}
              {class_html}
              {f'<h4>Functions</h4>{func_html}' if func_html else ''}
            </div><hr>"""

        # Dependency matrix
        all_names = sorted(m.name for m in self.modules)
        dep_header = "".join(f'<th class="rot"><div>{n[:10]}</div></th>' for n in all_names)
        dep_rows = ""
        for m in self.modules:
            cells = ""
            for t in all_names:
                hit = t in self.dep_graph.get(m.name, [])
                cells += f'<td class="{"dep-hit" if hit else ""}">{"●" if hit else ""}</td>'
            dep_rows += f'<tr><td class="dep-lbl">{m.name[:18]}</td>{cells}</tr>\n'

        # Coverage bar
        cov_pct = cov["coverage_pct"]
        cov_color = "#16a34a" if cov_pct >= 75 else "#d97706" if cov_pct >= 50 else "#dc2626"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>COMPASS Documentation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.2em; }}
  h3 {{ color: #1e40af; margin-top: 1.5em; }} h4 {{ color: #334155; margin-top: 1em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 2em; }}
  .doc {{ color: #475569; font-size: 0.9em; margin: 0.2em 0 0.6em; }}
  .doc-cell {{ color: #64748b; font-size: 0.9em; }}
  .meta-line {{ font-size: 0.85em; color: #64748b; margin: 0.2em 0; }}
  .sig {{ font-size: 0.85em; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
  a {{ color: #2563eb; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  code {{ font-size: 0.87em; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
  em {{ color: #7c3aed; font-style: normal; }}
  .tag {{ background: #dbeafe; color: #1e40af; padding: 1px 5px; border-radius: 3px; font-size: 0.78em; }}
  .class-card {{ background: #fff; border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;
                 border-radius: 4px; padding: 0.8em 1em; margin: 0.7em 0; }}
  .func-item {{ padding: 0.2em 0; font-size: 0.9em; }}
  .mtable {{ width: 100%; font-size: 0.85em; border-collapse: collapse; margin: 0.4em 0; }}
  .mtable th {{ background: #f8fafc; text-align: left; padding: 3px 6px; border-bottom: 1px solid #e2e8f0; }}
  .mtable td {{ padding: 3px 6px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
  table.idx {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  table.idx th {{ background: #f1f5f9; padding: 6px 10px; text-align: left; border-bottom: 2px solid #cbd5e1; }}
  table.idx td {{ padding: 5px 10px; border-bottom: 1px solid #e2e8f0; }}
  .cov-yes {{ color: #16a34a; font-weight: 700; }} .cov-no {{ color: #dc2626; font-weight: 700; }}
  .cov-bar {{ background: #e2e8f0; border-radius: 4px; height: 18px; width: 200px; display: inline-block; vertical-align: middle; }}
  .cov-fill {{ height: 100%; border-radius: 4px; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 1.5em 0; }}
  .dep-matrix {{ font-size: 0.65em; border-collapse: collapse; }}
  .dep-matrix th, .dep-matrix td {{ border: 1px solid #e2e8f0; padding: 1px 3px; text-align: center; }}
  .dep-lbl {{ text-align: left !important; font-weight: 500; white-space: nowrap; }}
  .dep-hit {{ background: #dbeafe; color: #1e40af; }}
  .rot {{ writing-mode: vertical-rl; transform: rotate(180deg); }} .rot div {{ max-height: 70px; overflow: hidden; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0; font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>COMPASS System Documentation</h1>
<div class="meta">
  {len(self.modules)} modules · {n_classes} classes · {n_funcs} functions · Generated {now}
</div>

<h2>Test Coverage</h2>
<p>{cov['tested_modules']}/{cov['total_modules']} modules tested ({cov_pct:.0f}%) · {cov['total_tests']} total test functions</p>
<div class="cov-bar"><div class="cov-fill" style="width:{cov_pct}%;background:{cov_color}"></div></div>
{f'<p class="meta-line">Untested: {", ".join(cov["untested_names"][:10])}{"..." if len(cov["untested_names"]) > 10 else ""}</p>' if cov["untested_names"] else ''}

<h2>Module Index</h2>
<table class="idx">
<thead><tr><th>Module</th><th>Classes</th><th>Functions</th><th>Lines</th><th>Tests</th></tr></thead>
<tbody>{index_rows}</tbody>
</table>

<h2>Module Reference</h2>
{sections}

<h2>Dependency Matrix</h2>
<p class="doc">● = row imports from column</p>
<div style="overflow-x:auto">
<table class="dep-matrix">
<thead><tr><th></th>{dep_header}</tr></thead>
<tbody>{dep_rows}</tbody>
</table>
</div>

<footer>Generated by <code>compass/auto_docs.py</code></footer>
</body></html>"""
        return html

    def _format_params(self, params: List[ParamInfo]) -> str:
        parts = []
        for p in params:
            s = html_mod.escape(p.name)
            if p.annotation:
                s += f": <em>{html_mod.escape(p.annotation)}</em>"
            if p.default:
                s += f" = {html_mod.escape(p.default)}"
            parts.append(s)
        return ", ".join(parts)

    def _format_sig(self, func: MethodInfo) -> str:
        params = self._format_params(func.params)
        ret = f" → <em>{html_mod.escape(func.returns)}</em>" if func.returns else ""
        return f"<strong>{html_mod.escape(func.name)}</strong>({params}){ret}"
