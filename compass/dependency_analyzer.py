"""
Import dependency analyzer for compass modules.

Builds a full dependency graph, detects circular imports, finds
tightly-coupled clusters, core modules, orphans, and computes
coupling metrics (Ce, Ca, instability).

All analysis is static (AST-based) — no module execution.
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModuleNode:
    name: str
    imports: List[str] = field(default_factory=list)      # modules this imports
    imported_by: List[str] = field(default_factory=list)   # modules that import this


@dataclass
class CouplingMetrics:
    module: str
    efferent: int          # Ce: number of modules this depends on
    afferent: int          # Ca: number of modules that depend on this
    instability: float     # Ce / (Ce + Ca)


@dataclass
class CircularImport:
    cycle: List[str]
    length: int


@dataclass
class ModuleCluster:
    cluster_id: int
    members: List[str]
    internal_edges: int
    cohesion: float        # internal / total edges


@dataclass
class DependencyReport:
    n_modules: int
    n_edges: int
    core_modules: List[Tuple[str, int]]    # (name, afferent_count) sorted desc
    orphan_modules: List[str]              # not imported by anything
    circular_imports: List[CircularImport]
    coupling: List[CouplingMetrics]
    clusters: List[ModuleCluster]
    suggestions: List[str]


class DependencyAnalyzer:
    """Static import dependency analyzer.

    Args:
        project_root: Root directory of the project.
        package: Package name to analyze (e.g. "compass").
    """

    def __init__(
        self, project_root: Optional[str] = None, package: str = "compass",
    ) -> None:
        self.root = Path(project_root) if project_root else Path.cwd()
        self.package = package
        self._graph: Dict[str, ModuleNode] = {}

    # ------------------------------------------------------------------
    # AST-based import extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_imports(filepath: Path, package: str) -> List[str]:
        """Extract compass-internal imports from a Python file."""
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError, OSError):
            return []

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(package):
                    # Extract the module name (e.g. compass.regime → regime)
                    parts = node.module.split(".")
                    if len(parts) >= 2:
                        imports.append(parts[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(package + "."):
                        parts = alias.name.split(".")
                        if len(parts) >= 2:
                            imports.append(parts[1])
        return list(set(imports))

    # ------------------------------------------------------------------
    # Build dependency graph
    # ------------------------------------------------------------------

    def build_graph(self) -> Dict[str, ModuleNode]:
        """Scan all modules and build the dependency graph."""
        pkg_dir = self.root / self.package
        if not pkg_dir.exists():
            return {}

        modules = sorted(
            f for f in pkg_dir.glob("*.py")
            if f.name != "__init__.py" and not f.name.startswith("_")
        )

        self._graph.clear()
        for f in modules:
            name = f.stem
            imports = self.extract_imports(f, self.package)
            self._graph[name] = ModuleNode(name=name, imports=imports)

        # Build reverse edges
        for name, node in self._graph.items():
            for imp in node.imports:
                if imp in self._graph:
                    self._graph[imp].imported_by.append(name)

        return dict(self._graph)

    # ------------------------------------------------------------------
    # Circular import detection
    # ------------------------------------------------------------------

    def detect_cycles(self) -> List[CircularImport]:
        """Find all circular import cycles via DFS."""
        if not self._graph:
            self.build_graph()

        visited: Set[str] = set()
        path: List[str] = []
        path_set: Set[str] = set()
        cycles: List[CircularImport] = []

        def dfs(node: str):
            if node in path_set:
                idx = path.index(node)
                cycle = path[idx:] + [node]
                cycles.append(CircularImport(cycle=cycle, length=len(cycle) - 1))
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for dep in self._graph.get(node, ModuleNode(node)).imports:
                if dep in self._graph:
                    dfs(dep)
            path.pop()
            path_set.discard(node)

        for name in self._graph:
            visited.clear()
            path.clear()
            path_set.clear()
            dfs(name)

        # Deduplicate cycles
        seen: Set[tuple] = set()
        unique: List[CircularImport] = []
        for c in cycles:
            key = tuple(sorted(c.cycle[:-1]))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # ------------------------------------------------------------------
    # Core modules and orphans
    # ------------------------------------------------------------------

    def core_modules(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Most-imported modules."""
        if not self._graph:
            self.build_graph()
        counts = [(name, len(node.imported_by)) for name, node in self._graph.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        return counts[:top_n]

    def orphan_modules(self) -> List[str]:
        """Modules not imported by any other compass module."""
        if not self._graph:
            self.build_graph()
        return sorted(name for name, node in self._graph.items()
                       if len(node.imported_by) == 0)

    # ------------------------------------------------------------------
    # Coupling metrics
    # ------------------------------------------------------------------

    def coupling_metrics(self) -> List[CouplingMetrics]:
        if not self._graph:
            self.build_graph()
        results: List[CouplingMetrics] = []
        for name, node in self._graph.items():
            ce = len(node.imports)
            ca = len(node.imported_by)
            instability = ce / (ce + ca) if (ce + ca) > 0 else 0.0
            results.append(CouplingMetrics(name, ce, ca, instability))
        results.sort(key=lambda c: c.instability, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Clustering (connected components)
    # ------------------------------------------------------------------

    def find_clusters(self) -> List[ModuleCluster]:
        """Find tightly-coupled module clusters via union-find."""
        if not self._graph:
            self.build_graph()

        parent: Dict[str, str] = {n: n for n in self._graph}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for name, node in self._graph.items():
            for imp in node.imports:
                if imp in self._graph:
                    union(name, imp)

        clusters_map: Dict[str, List[str]] = defaultdict(list)
        for name in self._graph:
            clusters_map[find(name)].append(name)

        results: List[ModuleCluster] = []
        for i, (root, members) in enumerate(sorted(clusters_map.items(), key=lambda x: -len(x[1]))):
            member_set = set(members)
            internal = sum(
                1 for m in members
                for imp in self._graph[m].imports if imp in member_set
            )
            total = sum(len(self._graph[m].imports) for m in members)
            cohesion = internal / total if total > 0 else 0.0
            results.append(ModuleCluster(i, sorted(members), internal, cohesion))
        return results

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def suggest_refactoring(self) -> List[str]:
        suggestions: List[str] = []
        cycles = self.detect_cycles()
        if cycles:
            suggestions.append(f"Fix {len(cycles)} circular import(s): "
                                + ", ".join(f"{' → '.join(c.cycle)}" for c in cycles[:3]))
        coupling = self.coupling_metrics()
        high_ce = [c for c in coupling if c.efferent > 10]
        if high_ce:
            suggestions.append(f"{len(high_ce)} modules have >10 dependencies — consider splitting")
        orphans = self.orphan_modules()
        if len(orphans) > 20:
            suggestions.append(f"{len(orphans)} orphan modules — consider consolidating or documenting")
        return suggestions

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(self) -> DependencyReport:
        graph = self.build_graph()
        cycles = self.detect_cycles()
        core = self.core_modules()
        orphans = self.orphan_modules()
        coupling = self.coupling_metrics()
        clusters = self.find_clusters()
        suggestions = self.suggest_refactoring()
        n_edges = sum(len(n.imports) for n in graph.values())

        return DependencyReport(
            n_modules=len(graph), n_edges=n_edges,
            core_modules=core, orphan_modules=orphans,
            circular_imports=cycles, coupling=coupling,
            clusters=clusters, suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, report: DependencyReport,
        output_path: str = "reports/dependency_analysis.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        core_rows = [f"<tr><td style='text-align:left'>{n}</td><td>{c}</td></tr>"
                      for n, c in report.core_modules[:15]]
        coupling_rows = [
            f"<tr><td style='text-align:left'>{c.module}</td>"
            f"<td>{c.efferent}</td><td>{c.afferent}</td>"
            f"<td>{c.instability:.2f}</td></tr>"
            for c in report.coupling[:20]
        ]
        cluster_rows = [
            f"<tr><td>{c.cluster_id}</td><td>{len(c.members)}</td>"
            f"<td>{c.internal_edges}</td><td>{c.cohesion:.2f}</td>"
            f"<td style='text-align:left'>{', '.join(c.members[:5])}{'...' if len(c.members) > 5 else ''}</td></tr>"
            for c in report.clusters[:10]
        ]
        cycle_html = ""
        if report.circular_imports:
            items = [f"<li>{' → '.join(c.cycle)}</li>" for c in report.circular_imports[:10]]
            cycle_html = f"<h2>Circular Imports ({len(report.circular_imports)})</h2><ul>{''.join(items)}</ul>"
        sugg_html = ""
        if report.suggestions:
            items = [f"<li>{s}</li>" for s in report.suggestions]
            sugg_html = f"<h2>Refactoring Suggestions</h2><ul>{''.join(items)}</ul>"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dependency Analysis</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Dependency Analysis Report</h1>
<div class="summary">
<p><strong>Modules:</strong> {report.n_modules} | <strong>Edges:</strong> {report.n_edges}
| <strong>Cycles:</strong> {len(report.circular_imports)}
| <strong>Orphans:</strong> {len(report.orphan_modules)}
| <strong>Clusters:</strong> {len(report.clusters)}</p>
</div>
<h2>Core Modules (most imported)</h2>
<table><tr><th style='text-align:left'>Module</th><th>Imported By</th></tr>
{''.join(core_rows)}</table>
<h2>Coupling Metrics</h2>
<table><tr><th style='text-align:left'>Module</th><th>Ce (out)</th><th>Ca (in)</th><th>Instability</th></tr>
{''.join(coupling_rows)}</table>
<h2>Clusters</h2>
<table><tr><th>ID</th><th>Size</th><th>Internal</th><th>Cohesion</th><th style='text-align:left'>Members</th></tr>
{''.join(cluster_rows)}</table>
{cycle_html}
{sugg_html}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
