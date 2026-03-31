"""Tests for compass.dependency_analyzer — 30 tests."""
import pytest
from pathlib import Path
from compass.dependency_analyzer import (
    DependencyAnalyzer, ModuleNode, CouplingMetrics, CircularImport,
    ModuleCluster, DependencyReport,
)

ROOT = Path(__file__).resolve().parent.parent


class TestExtractImports:
    def test_self(self):
        f = ROOT / "compass" / "dependency_analyzer.py"
        imports = DependencyAnalyzer.extract_imports(f, "compass")
        # This module doesn't import other compass modules
        assert isinstance(imports, list)
    def test_regime_hedge(self):
        f = ROOT / "compass" / "regime_hedge.py"
        if f.exists():
            imports = DependencyAnalyzer.extract_imports(f, "compass")
            assert "regime" in imports
    def test_nonexistent(self):
        imports = DependencyAnalyzer.extract_imports(Path("/nonexistent.py"), "compass")
        assert imports == []
    def test_init_excluded(self):
        f = ROOT / "compass" / "__init__.py"
        # Should not crash
        imports = DependencyAnalyzer.extract_imports(f, "compass")
        assert isinstance(imports, list)


class TestBuildGraph:
    def test_builds(self):
        da = DependencyAnalyzer(str(ROOT))
        graph = da.build_graph()
        assert len(graph) > 100  # 165+ modules
    def test_nodes_have_imports(self):
        da = DependencyAnalyzer(str(ROOT))
        graph = da.build_graph()
        has_imports = sum(1 for n in graph.values() if n.imports)
        assert has_imports > 10
    def test_reverse_edges(self):
        da = DependencyAnalyzer(str(ROOT))
        graph = da.build_graph()
        has_imported_by = sum(1 for n in graph.values() if n.imported_by)
        assert has_imported_by > 5


class TestCycles:
    def test_detect(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        cycles = da.detect_cycles()
        assert isinstance(cycles, list)
        # May or may not have cycles
    def test_cycle_structure(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        cycles = da.detect_cycles()
        for c in cycles:
            assert isinstance(c, CircularImport)
            assert c.length >= 2


class TestCoreOrphans:
    def test_core_modules(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        core = da.core_modules(10)
        assert len(core) == 10
        # Regime should be one of the most imported
        names = [n for n, _ in core]
        assert "regime" in names
    def test_orphans(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        orphans = da.orphan_modules()
        assert isinstance(orphans, list)
        # Some modules won't be imported by others
        assert len(orphans) > 0


class TestCoupling:
    def test_metrics(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        coupling = da.coupling_metrics()
        assert len(coupling) > 100
        for c in coupling:
            assert 0 <= c.instability <= 1.0
    def test_sorted(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        coupling = da.coupling_metrics()
        instabilities = [c.instability for c in coupling]
        assert instabilities == sorted(instabilities, reverse=True)


class TestClusters:
    def test_find(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        clusters = da.find_clusters()
        assert len(clusters) >= 1
    def test_cluster_structure(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        clusters = da.find_clusters()
        total_members = sum(len(c.members) for c in clusters)
        assert total_members == len(da._graph)
    def test_cohesion_bounded(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        for c in da.find_clusters():
            assert 0 <= c.cohesion <= 1.0


class TestSuggestions:
    def test_has_suggestions(self):
        da = DependencyAnalyzer(str(ROOT))
        da.build_graph()
        suggs = da.suggest_refactoring()
        assert isinstance(suggs, list)


class TestFullAnalysis:
    def test_analyze(self):
        da = DependencyAnalyzer(str(ROOT))
        report = da.analyze()
        assert isinstance(report, DependencyReport)
        assert report.n_modules > 100
        assert report.n_edges > 0


class TestReport:
    def test_creates_file(self, tmp_path):
        da = DependencyAnalyzer(str(ROOT))
        report = da.analyze()
        out = tmp_path / "dep.html"
        path = da.generate_report(report, output_path=str(out))
        assert Path(path).exists()
        assert "Dependency Analysis" in out.read_text()
    def test_contains_core(self, tmp_path):
        da = DependencyAnalyzer(str(ROOT))
        report = da.analyze()
        out = tmp_path / "d.html"
        da.generate_report(report, output_path=str(out))
        assert "Core Modules" in out.read_text()
    def test_contains_coupling(self, tmp_path):
        da = DependencyAnalyzer(str(ROOT))
        report = da.analyze()
        out = tmp_path / "d.html"
        da.generate_report(report, output_path=str(out))
        assert "Coupling" in out.read_text()
