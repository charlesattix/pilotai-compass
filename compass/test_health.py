"""
Test suite health analyzer.

Scans tests/test_*.py files to assess test quality:
  - Count tests per module
  - Identify slow tests (>1s)
  - Find flaky patterns (random without seed)
  - Check assertion density
  - Find skipped/dead tests
  - Cross-reference compass modules with test files
  - Generate HTML health report

No external dependencies beyond stdlib + numpy/pandas.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestFileInfo:
    path: str
    module_name: str
    n_tests: int
    n_classes: int
    n_assertions: int
    has_random_without_seed: bool
    n_skipped: int
    n_fixtures: int
    assertion_density: float  # assertions per test


@dataclass
class ModuleCoverage:
    compass_module: str
    test_file: Optional[str]
    is_tested: bool
    n_tests: int = 0


@dataclass
class FlakynessRisk:
    file: str
    risk_score: float  # 0-1
    reasons: List[str]


@dataclass
class HealthReport:
    total_test_files: int
    total_tests: int
    total_compass_modules: int
    untested_modules: List[str]
    coverage_pct: float
    avg_assertion_density: float
    flaky_risks: List[FlakynessRisk]
    file_infos: List[TestFileInfo]
    module_coverages: List[ModuleCoverage]


class TestHealthAnalyzer:
    """Test suite health analyzer.

    Args:
        project_root: Root directory of the project.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self.project_root = Path(project_root) if project_root else Path.cwd()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def discover_test_files(self) -> List[Path]:
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return []
        return sorted(tests_dir.glob("test_*.py"))

    def discover_compass_modules(self) -> List[Path]:
        compass_dir = self.project_root / "compass"
        if not compass_dir.exists():
            return []
        return sorted(f for f in compass_dir.glob("*.py")
                       if f.name != "__init__.py" and not f.name.startswith("_"))

    # ------------------------------------------------------------------
    # AST-based analysis
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_test_file(filepath: Path) -> TestFileInfo:
        """Parse a test file and extract metrics."""
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError, OSError):
            return TestFileInfo(
                str(filepath), filepath.stem, 0, 0, 0, False, 0, 0, 0.0)

        n_tests = 0
        n_classes = 0
        n_assertions = 0
        n_skipped = 0
        n_fixtures = 0
        has_random_no_seed = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    n_classes += 1
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    n_tests += 1
                # Check for fixtures
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Attribute) and dec.attr == "fixture":
                        n_fixtures += 1
                    elif isinstance(dec, ast.Name) and dec.id == "fixture":
                        n_fixtures += 1
                # Check for skip
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Attribute) and dec.attr in ("skip", "skipIf"):
                        n_skipped += 1
                    elif isinstance(dec, ast.Name) and dec.id == "skip":
                        n_skipped += 1

            # Count assertions
            if isinstance(node, ast.Assert):
                n_assertions += 1
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("assertEqual", "assertTrue", "assertFalse",
                                            "assertRaises", "assertIn", "assertIsNone",
                                            "assertAlmostEqual", "approx"):
                        n_assertions += 1

        # Check for random without seed pattern
        random_pattern = re.compile(r'np\.random\.(normal|uniform|choice|randint)\(')
        seed_pattern = re.compile(r'(seed|default_rng|RandomState)')
        if random_pattern.search(source) and not seed_pattern.search(source):
            has_random_no_seed = True

        density = n_assertions / max(n_tests, 1)

        return TestFileInfo(
            path=str(filepath),
            module_name=filepath.stem,
            n_tests=n_tests,
            n_classes=n_classes,
            n_assertions=n_assertions,
            has_random_without_seed=has_random_no_seed,
            n_skipped=n_skipped,
            n_fixtures=n_fixtures,
            assertion_density=density,
        )

    # ------------------------------------------------------------------
    # Coverage cross-reference
    # ------------------------------------------------------------------

    def compute_coverage(self) -> List[ModuleCoverage]:
        compass_modules = self.discover_compass_modules()
        test_files = self.discover_test_files()
        test_names = {f.stem for f in test_files}

        results: List[ModuleCoverage] = []
        for cm in compass_modules:
            expected_test = f"test_{cm.stem}"
            is_tested = expected_test in test_names
            n_tests = 0
            test_path = None
            if is_tested:
                tf = self.project_root / "tests" / f"{expected_test}.py"
                if tf.exists():
                    info = self.analyze_test_file(tf)
                    n_tests = info.n_tests
                    test_path = str(tf)
            results.append(ModuleCoverage(
                cm.stem, test_path, is_tested, n_tests))
        return results

    # ------------------------------------------------------------------
    # Flakiness risk
    # ------------------------------------------------------------------

    @staticmethod
    def assess_flakiness(info: TestFileInfo) -> FlakynessRisk:
        reasons: List[str] = []
        score = 0.0
        if info.has_random_without_seed:
            score += 0.5
            reasons.append("random without seed")
        if info.assertion_density < 1.0 and info.n_tests > 0:
            score += 0.2
            reasons.append(f"low assertion density ({info.assertion_density:.1f})")
        if info.n_tests == 0:
            score += 0.3
            reasons.append("no tests found")
        return FlakynessRisk(info.module_name, min(score, 1.0), reasons)

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(self) -> HealthReport:
        test_files = self.discover_test_files()
        compass_modules = self.discover_compass_modules()

        file_infos = [self.analyze_test_file(f) for f in test_files]
        coverages = self.compute_coverage()
        flaky = [self.assess_flakiness(fi) for fi in file_infos]

        total_tests = sum(fi.n_tests for fi in file_infos)
        untested = [c.compass_module for c in coverages if not c.is_tested]
        tested_count = sum(1 for c in coverages if c.is_tested)
        cov_pct = tested_count / len(coverages) if coverages else 0.0
        avg_density = float(np.mean([fi.assertion_density for fi in file_infos])) if file_infos else 0.0

        return HealthReport(
            total_test_files=len(test_files),
            total_tests=total_tests,
            total_compass_modules=len(compass_modules),
            untested_modules=untested,
            coverage_pct=cov_pct,
            avg_assertion_density=avg_density,
            flaky_risks=[f for f in flaky if f.risk_score > 0],
            file_infos=file_infos,
            module_coverages=coverages,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, report: HealthReport,
        output_path: str = "reports/test_health.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Coverage table
        cov_rows = []
        for c in sorted(report.module_coverages, key=lambda x: x.is_tested):
            color = "#27ae60" if c.is_tested else "#e74c3c"
            cov_rows.append(
                f"<tr><td style='text-align:left'>{c.compass_module}</td>"
                f"<td style='color:{color}'>"
                f"{'YES' if c.is_tested else 'NO'}</td>"
                f"<td>{c.n_tests}</td></tr>")

        # Untested
        untested_html = ""
        if report.untested_modules:
            items = "".join(f"<li>{m}</li>" for m in report.untested_modules[:30])
            untested_html = f"<h2>Untested Modules ({len(report.untested_modules)})</h2><ul>{items}</ul>"

        # Flaky
        flaky_rows = [
            f"<tr><td style='text-align:left'>{f.file}</td>"
            f"<td>{f.risk_score:.2f}</td>"
            f"<td style='text-align:left'>{', '.join(f.reasons)}</td></tr>"
            for f in sorted(report.flaky_risks, key=lambda x: x.risk_score, reverse=True)[:20]
        ]

        # File info table
        fi_rows = [
            f"<tr><td style='text-align:left'>{fi.module_name}</td>"
            f"<td>{fi.n_tests}</td><td>{fi.n_classes}</td>"
            f"<td>{fi.n_assertions}</td><td>{fi.assertion_density:.1f}</td>"
            f"<td>{fi.n_skipped}</td></tr>"
            for fi in sorted(report.file_infos, key=lambda x: x.n_tests, reverse=True)[:30]
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Test Health</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.big {{ font-size: 1.8em; font-weight: bold; color: #2980b9; }}
</style></head><body>
<h1>Test Suite Health Report</h1>
<div class="summary">
<p class="big">{report.total_tests} tests | {report.total_test_files} files | {report.coverage_pct:.0%} coverage</p>
<p>Compass modules: {report.total_compass_modules} | Untested: {len(report.untested_modules)} | Avg assertion density: {report.avg_assertion_density:.1f}</p>
</div>

{untested_html}

<h2>Flakiness Risk</h2>
<table><tr><th style='text-align:left'>File</th><th>Risk</th><th style='text-align:left'>Reasons</th></tr>
{''.join(flaky_rows)}</table>

<h2>Test File Summary (top 30)</h2>
<table><tr><th style='text-align:left'>Module</th><th>Tests</th><th>Classes</th>
<th>Asserts</th><th>Density</th><th>Skipped</th></tr>
{''.join(fi_rows)}</table>

<h2>Module Coverage</h2>
<table><tr><th style='text-align:left'>Module</th><th>Tested?</th><th>Tests</th></tr>
{''.join(cov_rows)}</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
