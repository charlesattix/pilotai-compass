"""Tests for compass.perf_benchmark — 18 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.perf_benchmark import PerfBenchmark, BenchmarkResult, BenchmarkSuite


class TestTimeFunction:
    def test_basic(self):
        r = PerfBenchmark.time_function(lambda: sum(range(100)), n=10)
        assert isinstance(r, BenchmarkResult)
        assert r.n_iterations == 10
        assert r.mean_ms > 0

    def test_percentiles_ordered(self):
        r = PerfBenchmark.time_function(lambda: np.ones(1000).sum(), n=50)
        assert r.p50_ms <= r.p95_ms <= r.p99_ms

    def test_total_ms(self):
        r = PerfBenchmark.time_function(lambda: None, n=20)
        assert r.total_ms >= 0


class TestBenchmarkCallable:
    def test_named(self):
        pb = PerfBenchmark(n_iterations=10)
        r = pb.benchmark_callable("test_op", lambda: 1 + 1)
        assert r.name == "test_op"
        assert r.n_iterations == 10

    def test_override_n(self):
        pb = PerfBenchmark(n_iterations=100)
        r = pb.benchmark_callable("fast", lambda: None, n=5)
        assert r.n_iterations == 5


class TestSuite:
    def test_run_suite(self):
        pb = PerfBenchmark(n_iterations=10, bottleneck_pct=0.30)
        suite = pb.run_suite({
            "fast": lambda: None,
            "slow": lambda: np.linalg.eigvalsh(np.eye(50)),
        })
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.results) == 2
        assert suite.bottleneck != ""

    def test_bottleneck_identified(self):
        pb = PerfBenchmark(n_iterations=10, bottleneck_pct=0.01)
        suite = pb.run_suite({"only": lambda: np.ones(100).sum()})
        assert suite.results[0].is_bottleneck

    def test_recommendations(self):
        pb = PerfBenchmark(n_iterations=10, bottleneck_pct=0.01)
        suite = pb.run_suite({"a": lambda: None, "b": lambda: np.linalg.det(np.eye(30))})
        assert isinstance(suite.recommendations, list)


class TestBuiltins:
    def test_numpy_ops(self):
        pb = PerfBenchmark(n_iterations=5)
        r = pb.benchmark_numpy_ops()
        assert r.mean_ms > 0

    def test_pandas_rolling(self):
        pb = PerfBenchmark(n_iterations=5)
        r = pb.benchmark_pandas_rolling()
        assert r.mean_ms > 0

    def test_covariance(self):
        pb = PerfBenchmark(n_iterations=5)
        r = pb.benchmark_covariance()
        assert r.mean_ms > 0

    def test_optimisation(self):
        pb = PerfBenchmark(n_iterations=5)
        r = pb.benchmark_optimisation()
        assert r.mean_ms > 0

    def test_builtin_suite(self):
        pb = PerfBenchmark(n_iterations=5)
        suite = pb.run_builtin_suite()
        assert len(suite.results) >= 3


class TestReport:
    def test_creates_file(self, tmp_path):
        pb = PerfBenchmark(n_iterations=5)
        suite = pb.run_builtin_suite()
        out = tmp_path / "perf.html"
        path = pb.generate_report(suite, output_path=str(out))
        assert Path(path).exists()
        assert "Performance Benchmark" in out.read_text()

    def test_contains_bottleneck(self, tmp_path):
        pb = PerfBenchmark(n_iterations=5)
        suite = pb.run_builtin_suite()
        out = tmp_path / "perf.html"
        pb.generate_report(suite, output_path=str(out))
        assert "Bottleneck" in out.read_text()
