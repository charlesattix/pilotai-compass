"""
Performance benchmarking for critical-path compass modules.

Measures execution latency of key pipeline stages, identifies
bottlenecks, and generates recommendations.

All benchmarks use synthetic data — no external dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    name: str
    n_iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float
    is_bottleneck: bool = False


@dataclass
class BenchmarkSuite:
    results: List[BenchmarkResult]
    total_ms: float
    bottleneck: str
    recommendations: List[str] = field(default_factory=list)


class PerfBenchmark:
    """Performance benchmark runner.

    Args:
        n_iterations: Default iterations per benchmark.
        bottleneck_pct: A stage is a bottleneck if it takes > this % of total.
    """

    def __init__(self, n_iterations: int = 100, bottleneck_pct: float = 0.30) -> None:
        self.n_iterations = n_iterations
        self.bottleneck_pct = bottleneck_pct

    @staticmethod
    def time_function(
        fn: Callable, n: int = 100, warmup: int = 2,
    ) -> BenchmarkResult:
        """Time a callable n times and return latency stats."""
        # Warmup
        for _ in range(warmup):
            fn()

        times: List[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        arr = np.array(times)
        return BenchmarkResult(
            name=getattr(fn, "__name__", "unknown"),
            n_iterations=n,
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            total_ms=float(arr.sum()),
        )

    def benchmark_callable(
        self, name: str, fn: Callable, n: Optional[int] = None,
    ) -> BenchmarkResult:
        """Named benchmark of a callable."""
        result = self.time_function(fn, n or self.n_iterations)
        result.name = name
        return result

    def run_suite(
        self, benchmarks: Dict[str, Callable],
    ) -> BenchmarkSuite:
        """Run a full benchmark suite."""
        results: List[BenchmarkResult] = []
        for name, fn in benchmarks.items():
            r = self.benchmark_callable(name, fn)
            results.append(r)

        total = sum(r.total_ms for r in results)
        # Identify bottleneck
        for r in results:
            if total > 0 and r.total_ms / total >= self.bottleneck_pct:
                r.is_bottleneck = True

        bottleneck = max(results, key=lambda r: r.total_ms).name if results else ""

        recs: List[str] = []
        for r in results:
            if r.is_bottleneck:
                recs.append(f"{r.name}: {r.mean_ms:.1f}ms avg — consider caching or vectorisation")
            if r.p99_ms > r.mean_ms * 5:
                recs.append(f"{r.name}: p99 ({r.p99_ms:.1f}ms) is {r.p99_ms / r.mean_ms:.0f}x mean — check for GC pauses")

        return BenchmarkSuite(
            results=results, total_ms=total,
            bottleneck=bottleneck, recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Built-in synthetic benchmarks
    # ------------------------------------------------------------------

    @staticmethod
    def _make_returns(n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2024-01-02", periods=n)
        return pd.DataFrame({
            f"asset_{i}": rng.normal(0.0003, 0.01, n) for i in range(5)
        }, index=idx)

    def benchmark_numpy_ops(self) -> BenchmarkResult:
        """Benchmark basic numpy operations (baseline)."""
        data = np.random.default_rng(42).normal(0, 1, (1000, 50))
        return self.benchmark_callable(
            "numpy_ops",
            lambda: np.linalg.eigvalsh(data.T @ data),
        )

    def benchmark_pandas_rolling(self) -> BenchmarkResult:
        """Benchmark pandas rolling computation."""
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 1000))
        return self.benchmark_callable(
            "pandas_rolling",
            lambda: s.rolling(20).mean(),
        )

    def benchmark_covariance(self) -> BenchmarkResult:
        """Benchmark covariance estimation."""
        ret = self._make_returns(500)
        return self.benchmark_callable(
            "covariance",
            lambda: ret.cov(),
        )

    def benchmark_optimisation(self) -> BenchmarkResult:
        """Benchmark scipy optimisation (portfolio-like)."""
        from scipy.optimize import minimize
        n = 10
        cov = np.eye(n) * 0.01
        mu = np.random.default_rng(42).normal(0.05, 0.02, n)
        def obj(w): return -(w @ mu) / np.sqrt(w @ cov @ w + 1e-8)
        x0 = np.ones(n) / n
        bounds = [(0, 0.3)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        return self.benchmark_callable(
            "optimisation",
            lambda: minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons),
        )

    def run_builtin_suite(self) -> BenchmarkSuite:
        """Run the built-in benchmark suite."""
        benchmarks = {
            "numpy_ops": lambda: np.linalg.eigvalsh(np.random.default_rng(42).normal(0, 1, (500, 20)).T @ np.random.default_rng(42).normal(0, 1, (500, 20))),
            "pandas_rolling": lambda: pd.Series(np.random.default_rng(42).normal(0, 1, 1000)).rolling(20).mean(),
            "covariance": lambda: self._make_returns(300).cov(),
        }
        return self.run_suite(benchmarks)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, suite: BenchmarkSuite,
        output_path: str = "reports/perf_benchmark.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in sorted(suite.results, key=lambda x: x.mean_ms, reverse=True):
            cls = " class='bottleneck'" if r.is_bottleneck else ""
            rows.append(
                f"<tr{cls}><td style='text-align:left'>{r.name}</td>"
                f"<td>{r.n_iterations}</td><td>{r.mean_ms:.2f}</td>"
                f"<td>{r.p50_ms:.2f}</td><td>{r.p95_ms:.2f}</td>"
                f"<td>{r.p99_ms:.2f}</td><td>{r.total_ms:.0f}</td></tr>")

        rec_html = ""
        if suite.recommendations:
            items = "".join(f"<li>{r}</li>" for r in suite.recommendations)
            rec_html = f"<h2>Recommendations</h2><ul>{items}</ul>"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Perf Benchmark</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr.bottleneck {{ background: #ffeaa7; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Performance Benchmark Report</h1>
<div class="summary">
<p><strong>Total:</strong> {suite.total_ms:.0f}ms |
   <strong>Bottleneck:</strong> {suite.bottleneck}</p>
</div>
<table><tr><th style='text-align:left'>Stage</th><th>Iters</th>
<th>Mean (ms)</th><th>P50</th><th>P95</th><th>P99</th><th>Total</th></tr>
{''.join(rows)}</table>
{rec_html}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
