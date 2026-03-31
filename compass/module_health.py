"""Module health checker for the compass package.

Discovers compass modules, validates imports, checks test coverage,
enforces naming conventions, and generates HTML health reports.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModuleStatus:
    """Health status of a single module."""

    name: str
    importable: bool
    has_tests: bool
    error: str = ""
    public_classes: list[str] = field(default_factory=list)
    public_functions: list[str] = field(default_factory=list)
    follows_conventions: bool = True


@dataclass
class HealthResult:
    """Aggregate health result across all discovered modules."""

    statuses: list[ModuleStatus] = field(default_factory=list)
    n_total: int = 0
    n_importable: int = 0
    n_with_tests: int = 0
    n_import_errors: int = 0
    modules_without_tests: list[str] = field(default_factory=list)
    import_errors: list[str] = field(default_factory=list)


class ModuleHealthChecker:
    """Discovers and validates compass modules."""

    def __init__(self, compass_dir: Path, tests_dir: Path) -> None:
        self.compass_dir = Path(compass_dir)
        self.tests_dir = Path(tests_dir)
        self._result: HealthResult | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_modules(self) -> list[Path]:
        """Return sorted list of .py files in compass_dir, excluding
        __init__.py and anything inside __pycache__."""
        modules: list[Path] = []
        for p in sorted(self.compass_dir.glob("*.py")):
            if p.name == "__init__.py":
                continue
            if "__pycache__" in str(p):
                continue
            modules.append(p)
        return modules

    # ------------------------------------------------------------------
    # Import attempt
    # ------------------------------------------------------------------

    def _try_import(self, module_path: Path) -> tuple[bool, str, object | None]:
        """Try to import *module_path* and return (success, error_msg, module).

        Uses ``importlib.util.spec_from_file_location`` so the module is
        loaded directly from the filesystem without requiring the parent
        package to already be on ``sys.path``.
        """
        # Build a dotted module name relative to compass_dir's parent so
        # that ``compass.foo`` works when the parent is on sys.path.
        parent = self.compass_dir.parent
        rel = module_path.relative_to(parent)
        dotted = ".".join(rel.with_suffix("").parts)

        # Ensure the parent directory is importable for transitive deps
        str_parent = str(parent)
        added = False
        if str_parent not in sys.path:
            sys.path.insert(0, str_parent)
            added = True

        try:
            # Make sure the compass package itself is importable
            pkg_name = self.compass_dir.name
            if pkg_name not in sys.modules:
                pkg_init = self.compass_dir / "__init__.py"
                if pkg_init.exists():
                    pkg_spec = importlib.util.spec_from_file_location(
                        pkg_name, pkg_init,
                        submodule_search_locations=[str(self.compass_dir)],
                    )
                    if pkg_spec and pkg_spec.loader:
                        pkg_mod = importlib.util.module_from_spec(pkg_spec)
                        sys.modules[pkg_name] = pkg_mod
                        pkg_spec.loader.exec_module(pkg_mod)

            spec = importlib.util.spec_from_file_location(dotted, module_path)
            if spec is None or spec.loader is None:
                return False, f"Could not create spec for {module_path}", None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
            return True, "", mod
        except Exception as exc:
            # Clean up partial module from sys.modules
            sys.modules.pop(dotted, None)
            return False, f"{exc.__class__.__name__}: {exc}", None
        finally:
            if added and str_parent in sys.path:
                sys.path.remove(str_parent)

    # ------------------------------------------------------------------
    # Test file matching
    # ------------------------------------------------------------------

    def _has_test_file(self, module_name: str) -> bool:
        """Check whether tests_dir contains ``test_{module_name}.py``."""
        return (self.tests_dir / f"test_{module_name}.py").exists()

    # ------------------------------------------------------------------
    # Public members
    # ------------------------------------------------------------------

    @staticmethod
    def _public_members(mod: object) -> tuple[list[str], list[str]]:
        """Return (public_classes, public_functions) of *mod*."""
        classes: list[str] = []
        functions: list[str] = []
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            # Only include members actually defined in the module
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isclass(obj):
                classes.append(name)
            elif inspect.isfunction(obj):
                functions.append(name)
        return sorted(classes), sorted(functions)

    # ------------------------------------------------------------------
    # Naming conventions
    # ------------------------------------------------------------------

    _SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
    _PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

    def _check_conventions(self, module_name: str, classes: list[str]) -> bool:
        """Return True when *module_name* is snake_case and all *classes*
        are PascalCase."""
        if not self._SNAKE_RE.match(module_name):
            return False
        for cls in classes:
            if not self._PASCAL_RE.match(cls):
                return False
        return True

    # ------------------------------------------------------------------
    # Circular import detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_circular_import(error_msg: str) -> bool:
        """Heuristic: does the import error mention circular imports?"""
        lower = error_msg.lower()
        return "circular" in lower or (
            "importerror" in lower and "cannot import name" in lower
            and "partially initialized" in lower
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check_all(self) -> HealthResult:
        """Run all checks and return an aggregated :class:`HealthResult`."""
        result = HealthResult()
        module_paths = self._discover_modules()
        result.n_total = len(module_paths)

        for path in module_paths:
            name = path.stem
            ok, err, mod = self._try_import(path)
            has_tests = self._has_test_file(name)

            classes: list[str] = []
            functions: list[str] = []
            if ok and mod is not None:
                classes, functions = self._public_members(mod)

            conventions = self._check_conventions(name, classes)

            status = ModuleStatus(
                name=name,
                importable=ok,
                has_tests=has_tests,
                error=err,
                public_classes=classes,
                public_functions=functions,
                follows_conventions=conventions,
            )
            result.statuses.append(status)

            if ok:
                result.n_importable += 1
            else:
                result.n_import_errors += 1
                result.import_errors.append(f"{name}: {err}")

            if has_tests:
                result.n_with_tests += 1
            else:
                result.modules_without_tests.append(name)

        self._result = result
        return result

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(self, result: HealthResult | None = None) -> str:
        """Return an HTML string summarising the health check."""
        if result is None:
            result = self._result
        if result is None:
            raise RuntimeError("No results available. Call check_all() first.")

        def _colour(status: ModuleStatus) -> str:
            if not status.importable:
                return "red"
            if not status.has_tests:
                return "yellow"
            return "green"

        def _badge(colour: str) -> str:
            labels = {"green": "OK", "yellow": "No Tests", "red": "Import Error"}
            return (
                f'<span style="background:{colour};color:#fff;'
                f'padding:2px 8px;border-radius:4px;">'
                f'{labels[colour]}</span>'
            )

        # --- summary cards ---
        pct_importable = (
            round(result.n_importable / result.n_total * 100)
            if result.n_total
            else 0
        )
        pct_tested = (
            round(result.n_with_tests / result.n_total * 100)
            if result.n_total
            else 0
        )

        cards_html = (
            '<div style="display:flex;gap:16px;margin-bottom:24px;">'
            f'<div style="background:#e8f5e9;padding:16px;border-radius:8px;flex:1;text-align:center;">'
            f'<div style="font-size:28px;font-weight:bold;">{result.n_total}</div>'
            f'<div>Total Modules</div></div>'
            f'<div style="background:#e3f2fd;padding:16px;border-radius:8px;flex:1;text-align:center;">'
            f'<div style="font-size:28px;font-weight:bold;">{result.n_importable} ({pct_importable}%)</div>'
            f'<div>Importable</div></div>'
            f'<div style="background:#fff3e0;padding:16px;border-radius:8px;flex:1;text-align:center;">'
            f'<div style="font-size:28px;font-weight:bold;">{result.n_with_tests} ({pct_tested}%)</div>'
            f'<div>With Tests</div></div>'
            f'<div style="background:#ffebee;padding:16px;border-radius:8px;flex:1;text-align:center;">'
            f'<div style="font-size:28px;font-weight:bold;">{result.n_import_errors}</div>'
            f'<div>Import Errors</div></div>'
            '</div>'
        )

        # --- per-module table ---
        rows: list[str] = []
        for s in result.statuses:
            colour = _colour(s)
            cls_str = ", ".join(s.public_classes) if s.public_classes else "-"
            fn_str = ", ".join(s.public_functions) if s.public_functions else "-"
            conv = "Yes" if s.follows_conventions else "No"
            rows.append(
                f"<tr>"
                f"<td>{s.name}</td>"
                f"<td>{_badge(colour)}</td>"
                f"<td>{conv}</td>"
                f"<td>{cls_str}</td>"
                f"<td>{fn_str}</td>"
                f"</tr>"
            )

        table_html = (
            '<table style="width:100%;border-collapse:collapse;" border="1" cellpadding="6">'
            "<thead><tr>"
            "<th>Module</th><th>Status</th><th>Conventions</th>"
            "<th>Public Classes</th><th>Public Functions</th>"
            "</tr></thead><tbody>"
            + "\n".join(rows)
            + "</tbody></table>"
        )

        # --- error details ---
        error_rows: list[str] = []
        for s in result.statuses:
            if s.error:
                safe_err = (
                    s.error.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                error_rows.append(
                    f"<tr><td>{s.name}</td><td><pre>{safe_err}</pre></td></tr>"
                )

        error_html = ""
        if error_rows:
            error_html = (
                "<h2>Import Error Details</h2>"
                '<table style="width:100%;border-collapse:collapse;" border="1" cellpadding="6">'
                "<thead><tr><th>Module</th><th>Error</th></tr></thead><tbody>"
                + "\n".join(error_rows)
                + "</tbody></table>"
            )

        html = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            "<title>Module Health Report</title>"
            "<style>body{font-family:sans-serif;margin:24px;}"
            "table{margin-top:12px;}th{background:#f5f5f5;}</style>"
            "</head><body>"
            "<h1>Module Health Report</h1>"
            + cards_html
            + "<h2>Module Status</h2>"
            + table_html
            + error_html
            + "</body></html>"
        )
        return html
