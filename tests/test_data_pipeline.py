"""Tests for compass.data_pipeline – automated data pipeline manager."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.data_pipeline import (
    DataPipeline,
    DataSource,
    DataVersion,
    DependencyNode,
    FeatureDef,
    PipelineResult,
    QualityCheck,
    SourceType,
    TaskResult,
    TaskStatus,
    build_dependency_graph,
    check_duplicates,
    check_missing,
    check_outliers,
    check_staleness,
    compute_checksum,
    make_version,
    retry_with_backoff,
    run_quality_checks,
    topological_sort,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "close": 450 + rng.randn(n).cumsum(),
        "volume": (1e6 + rng.randn(n) * 1e5).astype(int),
    }, index=idx)


def _good_fetch() -> pd.DataFrame:
    return _make_df()


def _bad_fetch() -> pd.DataFrame:
    raise ConnectionError("API unavailable")


_call_count = 0

def _flaky_fetch() -> pd.DataFrame:
    global _call_count
    _call_count += 1
    if _call_count % 3 != 0:
        raise ConnectionError("Transient error")
    return _make_df()


def _simple_feature(deps: dict) -> pd.DataFrame:
    df = list(deps.values())[0]
    return pd.DataFrame({"returns": df["close"].pct_change().dropna()})


def _make_sources() -> list:
    return [
        DataSource("prices", SourceType.YAHOO, ["SPY"], "daily", _good_fetch),
        DataSource("macro", SourceType.FRED, ["GDP"], "daily", _good_fetch),
    ]


def _make_features() -> list:
    return [
        FeatureDef("returns", ["prices"], _simple_feature, "Daily returns"),
    ]


# ── Quality checks ──────────────────────────────────────────────────────────
class TestCheckMissing:
    def test_clean_data_passes(self):
        assert check_missing(_make_df()).passed

    def test_many_missing_fails(self):
        df = _make_df()
        df.iloc[:20, 0] = np.nan
        assert not check_missing(df, max_pct=0.05).passed

    def test_empty_df_passes(self):
        assert check_missing(pd.DataFrame()).passed

    def test_count_reported(self):
        df = _make_df()
        df.iloc[0, 0] = np.nan
        c = check_missing(df)
        assert c.n_issues == 1


class TestCheckOutliers:
    def test_clean_data_passes(self):
        assert check_outliers(_make_df()).passed

    def test_extreme_value_fails(self):
        df = _make_df()
        df.iloc[0, 0] = 1e9
        assert not check_outliers(df, zscore_threshold=3.0).passed

    def test_no_numeric_passes(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        assert check_outliers(df).passed


class TestCheckStaleness:
    def test_fresh_passes(self):
        now = datetime.now(tz=timezone.utc).isoformat()
        assert check_staleness(now).passed

    def test_old_fails(self):
        old = (datetime.now(tz=timezone.utc) - timedelta(hours=48)).isoformat()
        assert not check_staleness(old, max_hours=24).passed

    def test_invalid_timestamp(self):
        assert not check_staleness("not-a-date").passed


class TestCheckDuplicates:
    def test_no_dups_passes(self):
        assert check_duplicates(_make_df()).passed

    def test_dups_detected(self):
        df = _make_df()
        df = pd.concat([df, df.iloc[:3]])
        assert not check_duplicates(df).passed


class TestRunQualityChecks:
    def test_returns_list(self):
        checks = run_quality_checks(_make_df())
        assert len(checks) >= 3

    def test_with_timestamp(self):
        now = datetime.now(tz=timezone.utc).isoformat()
        checks = run_quality_checks(_make_df(), last_updated=now)
        assert len(checks) == 4


# ── Versioning ──────────────────────────────────────────────────────────────
class TestVersioning:
    def test_checksum_deterministic(self):
        df = _make_df()
        assert compute_checksum(df) == compute_checksum(df)

    def test_different_data_different_checksum(self):
        assert compute_checksum(_make_df(seed=1)) != compute_checksum(_make_df(seed=2))

    def test_make_version_fields(self):
        v = make_version("test", _make_df())
        assert v.source_name == "test"
        assert v.n_rows == 100
        assert len(v.checksum) == 16
        assert len(v.last_updated) > 0


# ── Retry logic ─────────────────────────────────────────────────────────────
class TestRetry:
    def test_succeeds_first_try(self):
        result, retries = retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42
        assert retries == 0

    def test_retries_on_failure(self):
        global _call_count
        _call_count = 0
        result, retries = retry_with_backoff(_flaky_fetch, max_retries=5, initial_delay=0.001)
        assert isinstance(result, pd.DataFrame)
        assert retries > 0

    def test_raises_after_max(self):
        with pytest.raises(ConnectionError):
            retry_with_backoff(_bad_fetch, max_retries=2, initial_delay=0.001)


# ── Dependency graph ────────────────────────────────────────────────────────
class TestDependencyGraph:
    def test_build_graph(self):
        graph = build_dependency_graph(_make_sources(), _make_features())
        assert len(graph) == 3  # 2 sources + 1 feature

    def test_feature_depends_on_source(self):
        graph = build_dependency_graph(_make_sources(), _make_features())
        ret_node = next(n for n in graph if n.name == "returns")
        assert "prices" in ret_node.depends_on

    def test_source_has_dependent(self):
        graph = build_dependency_graph(_make_sources(), _make_features())
        prices_node = next(n for n in graph if n.name == "prices")
        assert "returns" in prices_node.dependents

    def test_topological_sort_sources_first(self):
        graph = build_dependency_graph(_make_sources(), _make_features())
        order = topological_sort(graph)
        prices_idx = order.index("prices")
        returns_idx = order.index("returns")
        assert prices_idx < returns_idx

    def test_empty_graph(self):
        assert topological_sort([]) == []


# ── Pipeline run ────────────────────────────────────────────────────────────
class TestPipelineRun:
    def test_returns_result(self):
        p = DataPipeline(sources=_make_sources(), features=_make_features())
        r = p.run()
        assert isinstance(r, PipelineResult)

    def test_all_tasks_succeed(self):
        p = DataPipeline(sources=_make_sources(), features=_make_features())
        r = p.run()
        assert r.n_succeeded == 3  # 2 sources + 1 feature
        assert r.n_failed == 0

    def test_failed_source(self):
        sources = [DataSource("bad", SourceType.CUSTOM, [], "daily", _bad_fetch)]
        p = DataPipeline(sources=sources, max_retries=1)
        r = p.run()
        assert r.n_failed == 1

    def test_data_stored(self):
        p = DataPipeline(sources=_make_sources())
        p.run()
        assert p.get_data("prices") is not None
        assert len(p.get_data("prices")) > 0

    def test_version_stored(self):
        p = DataPipeline(sources=_make_sources())
        p.run()
        v = p.get_version("prices")
        assert v is not None
        assert v.n_rows > 0

    def test_feature_computed(self):
        p = DataPipeline(sources=_make_sources(), features=_make_features())
        p.run()
        ret = p.get_data("returns")
        assert ret is not None
        assert "returns" in ret.columns

    def test_missing_dependency_fails(self):
        features = [FeatureDef("bad_feat", ["nonexistent"], _simple_feature)]
        p = DataPipeline(features=features)
        r = p.run()
        assert r.n_failed == 1

    def test_no_fetch_fn_fails(self):
        sources = [DataSource("nofn", SourceType.CUSTOM)]
        p = DataPipeline(sources=sources)
        r = p.run()
        assert r.n_failed == 1

    def test_generated_at(self):
        p = DataPipeline(sources=_make_sources())
        r = p.run()
        assert len(r.generated_at) > 0

    def test_quality_checks_run(self):
        p = DataPipeline(sources=_make_sources())
        r = p.run()
        assert len(r.quality_checks) > 0

    def test_total_rows(self):
        p = DataPipeline(sources=_make_sources())
        r = p.run()
        assert r.total_rows > 0


# ── Incremental updates ────────────────────────────────────────────────────
class TestIncremental:
    def test_incremental_no_duplicates(self):
        p = DataPipeline(sources=_make_sources())
        p.run(incremental=True)
        first_len = len(p.get_data("prices"))
        p.run(incremental=True)
        second_len = len(p.get_data("prices"))
        assert second_len == first_len  # no new data → same length


# ── Add source/feature ──────────────────────────────────────────────────────
class TestAddSourceFeature:
    def test_add_source(self):
        p = DataPipeline()
        p.add_source(DataSource("x", SourceType.CUSTOM, fetch_fn=_good_fetch))
        assert len(p.sources) == 1

    def test_add_feature(self):
        p = DataPipeline()
        p.add_feature(FeatureDef("f", ["x"], _simple_feature))
        assert len(p.features) == 1


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DataPipeline(sources=_make_sources(), features=_make_features())
            r = p.run()
            path = p.generate_report(r, output_path=Path(tmp) / "dp.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DataPipeline(sources=_make_sources(), features=_make_features())
            r = p.run()
            path = p.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Data Pipeline" in html
            assert "Pipeline Tasks" in html
            assert "Quality" in html
            assert "Versions" in html
            assert "Dependency" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DataPipeline(sources=_make_sources())
            r = p.run()
            path = p.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_data_source(self):
        s = DataSource("x", SourceType.ALPACA, ["SPY"])
        assert s.source_type == SourceType.ALPACA

    def test_quality_check(self):
        c = QualityCheck("test", True, "ok", 0)
        assert c.passed

    def test_data_version(self):
        v = DataVersion("x", "abc123", 100, 5, "2024-01-01")
        assert v.n_rows == 100

    def test_task_result(self):
        t = TaskResult("fetch", TaskStatus.SUCCESS)
        assert t.status == TaskStatus.SUCCESS

    def test_pipeline_result_defaults(self):
        r = PipelineResult()
        assert r.tasks == []
        assert r.n_succeeded == 0
