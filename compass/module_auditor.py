"""
Meta-audit of all compass modules for quality and completeness.

Scans compass/*.py, counts lines/classes/functions/tests, checks HTML
reports, verifies imports, detects duplicates, and generates a
comprehensive HTML report with catalog, dependency graph, coverage
gaps, and quality scores.

Usage::

    from compass.module_auditor import ModuleAuditor
    auditor = ModuleAuditor()
    results = auditor.audit()
    auditor.generate_report()
"""

from __future__ import annotations

import ast
import base64
import io
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
COMPASS_DIR = ROOT / "compass"
TESTS_DIR = ROOT / "tests"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "module_audit.html"


@dataclass
class ModuleInfo:
    name: str
    path: str
    lines: int
    classes: int
    functions: int
    has_docstring: bool
    has_test_file: bool
    test_count: int
    has_html_report: bool
    imports_compass: List[str]
    imported_by: List[str]
    import_works: bool
    quality_score: float       # 0-100
    grade: str


@dataclass
class DuplicateGroup:
    modules: List[str]
    overlap_type: str          # "name_similar", "function_overlap"
    similarity: float
    description: str


@dataclass
class AuditSummary:
    total_modules: int
    total_lines: int
    total_classes: int
    total_functions: int
    modules_with_tests: int
    modules_without_tests: int
    total_tests: int
    modules_with_reports: int
    orphan_modules: List[str]
    avg_quality: float
    duplicates: List[DuplicateGroup]
    grade_distribution: Dict[str, int]


class ModuleAuditor:
    """Audit all compass modules."""

    def __init__(self, compass_dir: Optional[Path] = None, skip_imports: bool = False) -> None:
        self.compass_dir = compass_dir or COMPASS_DIR
        self.tests_dir = self.compass_dir.parent / "tests"
        self.reports_dir = self.compass_dir.parent / "reports"
        self._skip_imports = skip_imports
        self.modules: List[ModuleInfo] = []
        self.summary: Optional[AuditSummary] = None

    def audit(self) -> Dict[str, Any]:
        py_files = sorted(self.compass_dir.glob("*.py"))
        all_info: List[ModuleInfo] = []
        # First pass: collect basic info
        for f in py_files:
            if f.name == "__init__.py":
                continue
            info = self._analyze_module(f)
            all_info.append(info)

        # Second pass: imported_by
        name_set = {m.name for m in all_info}
        for m in all_info:
            for other in all_info:
                if m.name != other.name and m.name in other.imports_compass:
                    m.imported_by.append(other.name)

        # Quality scores
        for m in all_info:
            m.quality_score = self._score(m)
            m.grade = self._grade(m.quality_score)

        self.modules = all_info
        duplicates = self._detect_duplicates()
        orphans = [m.name for m in all_info if not m.imported_by and m.name != "__init__"]

        grade_dist: Dict[str, int] = {}
        for m in all_info:
            grade_dist[m.grade] = grade_dist.get(m.grade, 0) + 1

        self.summary = AuditSummary(
            total_modules=len(all_info),
            total_lines=sum(m.lines for m in all_info),
            total_classes=sum(m.classes for m in all_info),
            total_functions=sum(m.functions for m in all_info),
            modules_with_tests=sum(1 for m in all_info if m.has_test_file),
            modules_without_tests=sum(1 for m in all_info if not m.has_test_file),
            total_tests=sum(m.test_count for m in all_info),
            modules_with_reports=sum(1 for m in all_info if m.has_html_report),
            orphan_modules=orphans[:30],
            avg_quality=float(np.mean([m.quality_score for m in all_info])) if all_info else 0,
            duplicates=duplicates,
            grade_distribution=grade_dist,
        )
        return {
            "modules": self.modules,
            "summary": self.summary,
        }

    def _analyze_module(self, path: Path) -> ModuleInfo:
        name = path.stem
        try:
            source = path.read_text(errors="replace")
        except Exception:
            source = ""

        lines = source.count("\n") + 1 if source else 0

        # AST analysis
        classes = functions = 0
        has_doc = False
        imports_compass: List[str] = []
        try:
            tree = ast.parse(source)
            has_doc = (isinstance(tree.body[0], ast.Expr) and
                       isinstance(tree.body[0].value, (ast.Constant, ast.Str))) if tree.body else False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    functions += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module and "compass." in (node.module or ""):
                        dep = node.module.split("compass.")[-1].split(".")[0]
                        if dep and dep != name:
                            imports_compass.append(dep)
        except SyntaxError:
            pass

        # Test file
        test_path = self.tests_dir / f"test_{name}.py"
        has_test = test_path.exists()
        test_count = 0
        if has_test:
            try:
                test_src = test_path.read_text(errors="replace")
                test_count = len(re.findall(r"def test_", test_src))
            except Exception:
                pass

        # HTML report
        has_report = (self.reports_dir / f"{name}.html").exists()

        # Import check
        import_works = self._check_import(name)

        return ModuleInfo(
            name=name, path=str(path), lines=lines,
            classes=classes, functions=functions,
            has_docstring=has_doc, has_test_file=has_test,
            test_count=test_count, has_html_report=has_report,
            imports_compass=imports_compass, imported_by=[],
            import_works=import_works, quality_score=0, grade="F",
        )

    def _check_import(self, name: str) -> bool:
        if self._skip_imports:
            return True
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import compass.{name}"],
                capture_output=True, timeout=5,
                cwd=str(ROOT),
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    @staticmethod
    def _score(m: ModuleInfo) -> float:
        s = 0.0
        s += 20 if m.has_test_file else 0
        s += min(20, m.test_count * 0.6)
        s += 15 if m.has_docstring else 0
        s += 10 if m.import_works else 0
        s += 10 if m.has_html_report else 0
        s += min(15, m.lines / 50)  # longer = more substance, up to 15
        s += min(10, m.classes * 2 + m.functions * 0.5)
        return min(100, s)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _detect_duplicates(self) -> List[DuplicateGroup]:
        groups: List[DuplicateGroup] = []
        names = [m.name for m in self.modules]
        seen: Set[Tuple[str, str]] = set()

        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i >= j:
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                # Name similarity
                tokens_a = set(a.replace("_", " ").split())
                tokens_b = set(b.replace("_", " ").split())
                if tokens_a and tokens_b:
                    overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
                    if overlap >= 0.5 and len(tokens_a & tokens_b) >= 2:
                        seen.add(key)
                        groups.append(DuplicateGroup(
                            modules=[a, b], overlap_type="name_similar",
                            similarity=overlap,
                            description=f"Shared tokens: {tokens_a & tokens_b}",
                        ))
        return sorted(groups, key=lambda g: -g.similarity)[:20]

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.summary is None:
            self.audit()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        return {
            "grades": self._chart_grades(),
            "coverage": self._chart_coverage(),
        }

    def _chart_grades(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.summary:
            return ""
        d = self.summary.grade_distribution
        grades = ["A", "B", "C", "D", "F"]
        vals = [d.get(g, 0) for g in grades]
        colors = ["#16a34a", "#22c55e", "#f59e0b", "#f97316", "#dc2626"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(grades, vals, color=colors, alpha=0.85)
        ax.set_ylabel("Modules"); ax.set_title("Quality Grade Distribution", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_coverage(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.summary:
            return ""
        labels = ["With Tests", "Without Tests"]
        vals = [self.summary.modules_with_tests, self.summary.modules_without_tests]
        colors = ["#16a34a", "#dc2626"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title("Test Coverage", fontsize=11); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = self.summary

        # Module catalog
        rows = ""
        for m in sorted(self.modules, key=lambda x: -x.quality_score):
            g_cls = "good" if m.grade in ("A", "B") else "bad" if m.grade in ("D", "F") else ""
            test_cls = "good" if m.has_test_file else "bad"
            rows += (f'<tr><td>{m.name}</td><td>{m.lines}</td><td>{m.classes}</td>'
                     f'<td>{m.functions}</td><td class="{test_cls}">{m.test_count}</td>'
                     f'<td>{"Y" if m.has_html_report else ""}</td>'
                     f'<td>{"Y" if m.import_works else "<span class=bad>N</span>"}</td>'
                     f'<td>{len(m.imports_compass)}</td>'
                     f'<td class="{g_cls}">{m.grade} ({m.quality_score:.0f})</td></tr>\n')

        # Gaps
        gap_rows = ""
        for m in self.modules:
            if not m.has_test_file:
                gap_rows += f'<tr><td>{m.name}</td><td>{m.lines} lines</td><td>No test file</td></tr>\n'
        if not gap_rows:
            gap_rows = '<tr><td colspan="3" style="text-align:center;color:#64748b">All modules have tests!</td></tr>'

        # Duplicates
        dup_rows = ""
        for d in (s.duplicates if s else []):
            dup_rows += f'<tr><td>{" / ".join(d.modules)}</td><td>{d.overlap_type}</td><td>{d.similarity:.0%}</td><td>{d.description}</td></tr>\n'
        if not dup_rows:
            dup_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No duplicates</td></tr>'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Compass Module Audit</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.82em; }}
  th {{ background:#f1f5f9; padding:6px 8px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:5px 8px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Compass Module Audit Report</h1>
<div class="meta">{s.total_modules if s else 0} modules &middot; {s.total_lines if s else 0:,} lines &middot; {s.total_tests if s else 0} tests &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value">{s.total_modules if s else 0}</div><div class="label">Modules</div></div>
  <div class="kpi"><div class="value">{s.total_lines if s else 0:,}</div><div class="label">Lines of Code</div></div>
  <div class="kpi"><div class="value">{s.total_tests if s else 0}</div><div class="label">Total Tests</div></div>
  <div class="kpi"><div class="value good">{s.modules_with_tests if s else 0}</div><div class="label">With Tests</div></div>
  <div class="kpi"><div class="value bad">{s.modules_without_tests if s else 0}</div><div class="label">No Tests</div></div>
  <div class="kpi"><div class="value">{s.avg_quality if s else 0:.0f}</div><div class="label">Avg Quality</div></div>
</div>
<h2>1. Grade Distribution</h2>{_img("grades")}
<h2>2. Test Coverage</h2>{_img("coverage")}
<h2>3. Module Catalog</h2>
<table><thead><tr><th>Module</th><th>Lines</th><th>Classes</th><th>Funcs</th><th>Tests</th><th>Report</th><th>Import</th><th>Deps</th><th>Grade</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>4. Coverage Gaps</h2>
<table><thead><tr><th>Module</th><th>Size</th><th>Issue</th></tr></thead><tbody>{gap_rows}</tbody></table>
<h2>5. Potential Duplicates</h2>
<table><thead><tr><th>Modules</th><th>Type</th><th>Similarity</th><th>Details</th></tr></thead><tbody>{dup_rows}</tbody></table>
<footer>Generated by <code>compass/module_auditor.py</code></footer>
</body></html>"""
        return html
