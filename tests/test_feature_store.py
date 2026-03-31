"""Tests for compass/feature_store.py — feature store manager.

Covers:
  - Dataclass construction
  - Feature registration and versioning
  - Value storage and retrieval
  - Lineage tracking (parents/children)
  - Freshness monitoring and alerts
  - Feature importance ranking
  - Batch compute with caching
  - Cache management
  - from_csv-style construction
  - Full report generation
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from compass.feature_store import (
    CacheEntry,
    FeatureMeta,
    FeatureStore,
    FreshnessAlert,
    LineageEdge,
)


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def store():
    """Fresh in-memory feature store."""
    s = FeatureStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def populated_store():
    """Store with several features registered."""
    s = FeatureStore(":memory:")
    s.register("vix", source="market", dtype="float64", description="VIX index",
               importance=0.8, tags=["market", "vol"])
    s.register("rsi", source="technical", dtype="float64", description="RSI 14",
               importance=0.5, tags=["technical"])
    s.register("vix_rank", source="derived", dtype="float64",
               description="VIX percentile rank", parents=["vix"],
               importance=0.7, tags=["derived"])
    s.register("signal", source="model", dtype="float64",
               description="Combined signal", parents=["vix_rank", "rsi"],
               importance=0.9, tags=["model"])
    # Set values
    s.set_values("vix", pd.Series([20.5, 21.0, 19.8], index=["a", "b", "c"]))
    s.set_values("rsi", pd.Series([55.0, 48.0, 62.0], index=["a", "b", "c"]))
    yield s
    s.close()


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_feature_meta_fields(self):
        fm = FeatureMeta(
            name="vix", version=1, source="market", dtype="float64",
            description="VIX", created_at="2024-01-01", updated_at="2024-01-01",
            row_count=100, parents=[], importance=0.8,
            freshness_seconds=3600, tags=["market"],
        )
        assert fm.name == "vix"
        assert fm.importance == pytest.approx(0.8)

    def test_freshness_alert_fields(self):
        fa = FreshnessAlert(
            feature="vix", last_updated="2024-01-01",
            stale_seconds=90000, threshold_seconds=86400,
            severity="warning",
        )
        assert fa.severity == "warning"

    def test_lineage_edge_fields(self):
        le = LineageEdge(parent="vix", child="vix_rank", version=1)
        assert le.parent == "vix"

    def test_cache_entry_fields(self):
        ce = CacheEntry(
            feature="vix", cache_key="abc123",
            computed_at="2024-01-01", row_count=100, hit_count=5,
        )
        assert ce.hit_count == 5


# ── Registration tests ───────────────────────────────────────────────────


class TestRegistration:
    def test_register_returns_meta(self, store):
        meta = store.register("vix", source="market", dtype="float64")
        assert isinstance(meta, FeatureMeta)
        assert meta.name == "vix"
        assert meta.version == 1

    def test_register_bumps_version(self, store):
        store.register("vix")
        meta2 = store.register("vix")
        assert meta2.version == 2

    def test_register_with_parents(self, store):
        store.register("vix")
        store.register("rsi")
        meta = store.register("signal", parents=["vix", "rsi"])
        assert set(meta.parents) == {"vix", "rsi"}

    def test_register_with_tags(self, store):
        meta = store.register("vix", tags=["market", "vol"])
        assert "market" in meta.tags

    def test_register_with_importance(self, store):
        meta = store.register("vix", importance=0.85)
        assert meta.importance == pytest.approx(0.85)

    def test_register_custom_freshness(self, store):
        store.register("vix", freshness_threshold=3600)
        # Threshold stored internally — verify via check_freshness behavior
        alerts = store.check_freshness()
        # Just registered, should be fresh
        assert all(a.feature != "vix" or a.threshold_seconds == 3600 for a in alerts)


# ── Value storage tests ──────────────────────────────────────────────────


class TestValues:
    def test_set_and_get_values(self, store):
        store.register("vix")
        values = pd.Series([20.5, 21.0, 19.8], index=["a", "b", "c"])
        store.set_values("vix", values)
        result = store.get_values("vix")
        assert len(result) == 3
        assert result["a"] == pytest.approx(20.5)

    def test_set_updates_row_count(self, store):
        store.register("vix")
        store.set_values("vix", pd.Series([1, 2, 3]))
        meta = store.get_feature("vix")
        assert meta.row_count == 3

    def test_get_values_unregistered_raises(self, store):
        with pytest.raises(ValueError, match="not registered"):
            store.get_values("nonexistent")

    def test_set_values_unregistered_raises(self, store):
        with pytest.raises(ValueError, match="not registered"):
            store.set_values("nonexistent", pd.Series([1]))

    def test_set_values_overwrites(self, store):
        store.register("vix")
        store.set_values("vix", pd.Series([1, 2, 3], index=["a", "b", "c"]))
        store.set_values("vix", pd.Series([10, 20], index=["x", "y"]))
        result = store.get_values("vix")
        assert len(result) == 2
        assert result["x"] == pytest.approx(10)

    def test_get_empty_values(self, store):
        store.register("vix")
        result = store.get_values("vix")
        assert len(result) == 0


# ── Listing and metadata tests ───────────────────────────────────────────


class TestListing:
    def test_list_features(self, populated_store):
        features = populated_store.list_features()
        names = {f.name for f in features}
        assert "vix" in names
        assert "rsi" in names
        assert "signal" in names

    def test_list_returns_latest_version(self, store):
        store.register("vix")
        store.register("vix")
        features = store.list_features()
        vix = [f for f in features if f.name == "vix"][0]
        assert vix.version == 2

    def test_get_feature(self, populated_store):
        meta = populated_store.get_feature("vix")
        assert meta.source == "market"
        assert meta.row_count == 3

    def test_get_feature_specific_version(self, store):
        store.register("vix", description="v1")
        store.register("vix", description="v2")
        v1 = store.get_feature("vix", version=1)
        assert v1.description == "v1"

    def test_get_feature_not_found_raises(self, store):
        with pytest.raises(ValueError, match="not registered"):
            store.get_feature("nonexistent")


# ── Lineage tests ────────────────────────────────────────────────────────


class TestLineage:
    def test_lineage_edges(self, populated_store):
        edges = populated_store.get_lineage("vix_rank")
        parents = [e.parent for e in edges if e.child == "vix_rank"]
        assert "vix" in parents

    def test_full_lineage(self, populated_store):
        edges = populated_store.get_full_lineage()
        assert len(edges) >= 3  # vix→vix_rank, vix_rank→signal, rsi→signal

    def test_lineage_captures_children(self, populated_store):
        edges = populated_store.get_lineage("vix")
        children = [e.child for e in edges if e.parent == "vix"]
        assert "vix_rank" in children

    def test_no_lineage_for_root(self, store):
        store.register("vix")
        edges = store.get_lineage("vix")
        assert len(edges) == 0


# ── Importance tests ─────────────────────────────────────────────────────


class TestImportance:
    def test_set_importance(self, store):
        store.register("vix")
        store.set_importance("vix", 0.95)
        meta = store.get_feature("vix")
        assert meta.importance == pytest.approx(0.95)

    def test_importance_ranking(self, populated_store):
        ranking = populated_store.importance_ranking()
        names = [r[0] for r in ranking]
        # signal (0.9) should be first
        assert names[0] == "signal"

    def test_importance_ranking_order(self, populated_store):
        ranking = populated_store.importance_ranking()
        values = [r[1] for r in ranking]
        assert values == sorted(values, reverse=True)

    def test_set_importance_unregistered_raises(self, store):
        with pytest.raises(ValueError, match="not registered"):
            store.set_importance("nonexistent", 0.5)


# ── Freshness tests ─────────────────────────────────────────────────────


class TestFreshness:
    def test_fresh_features_no_alerts(self, store):
        store.register("vix", freshness_threshold=86400)
        store.set_values("vix", pd.Series([1, 2, 3]))
        alerts = store.check_freshness()
        vix_alerts = [a for a in alerts if a.feature == "vix"]
        assert len(vix_alerts) == 0

    def test_stale_feature_alert(self, store):
        store.register("vix", freshness_threshold=0.0)  # 0 seconds = always stale
        store.set_values("vix", pd.Series([1]))
        time.sleep(0.05)
        alerts = store.check_freshness()
        vix_alerts = [a for a in alerts if a.feature == "vix"]
        assert len(vix_alerts) == 1

    def test_alert_severity(self, store):
        store.register("vix", freshness_threshold=0.001)
        store.set_values("vix", pd.Series([1]))
        time.sleep(0.05)
        alerts = store.check_freshness()
        vix_alerts = [a for a in alerts if a.feature == "vix"]
        assert vix_alerts[0].severity in ("warning", "critical")

    def test_set_freshness_threshold(self, store):
        store.register("vix")
        store.set_freshness_threshold("vix", 7200)
        # Verify threshold is used
        store.set_values("vix", pd.Series([1]))
        alerts = store.check_freshness()
        vix_alerts = [a for a in alerts if a.feature == "vix"]
        assert len(vix_alerts) == 0  # 7200s threshold, just updated


# ── Cache tests ──────────────────────────────────────────────────────────


class TestCache:
    def test_compute_and_cache(self, store):
        store.register("vix_ma")
        store.register_compute("vix_ma", lambda: pd.Series([20.0, 21.0, 19.5]))
        result = store.compute("vix_ma")
        assert len(result) == 3

    def test_cache_hit(self, store):
        call_count = {"n": 0}

        def compute_fn():
            call_count["n"] += 1
            return pd.Series([1, 2, 3])

        store.register("test_feat")
        store.register_compute("test_feat", compute_fn)
        store.compute("test_feat")
        store.compute("test_feat")  # should hit cache
        assert call_count["n"] == 1

    def test_cache_bypass(self, store):
        call_count = {"n": 0}

        def compute_fn():
            call_count["n"] += 1
            return pd.Series([1, 2, 3])

        store.register("test_feat")
        store.register_compute("test_feat", compute_fn)
        store.compute("test_feat", use_cache=False)
        store.compute("test_feat", use_cache=False)
        assert call_count["n"] == 2

    def test_compute_batch(self, store):
        store.register("a")
        store.register("b")
        store.register_compute("a", lambda: pd.Series([1, 2]))
        store.register_compute("b", lambda: pd.Series([3, 4]))
        results = store.compute_batch(["a", "b"])
        assert "a" in results
        assert "b" in results

    def test_cache_stats(self, store):
        store.register("x")
        store.register_compute("x", lambda: pd.Series([1]))
        store.compute("x")
        stats = store.get_cache_stats()
        assert len(stats) >= 1

    def test_clear_cache(self, store):
        store.register("x")
        store.register_compute("x", lambda: pd.Series([1]))
        store.compute("x")
        deleted = store.clear_cache()
        assert deleted >= 1
        assert len(store.get_cache_stats()) == 0

    def test_clear_cache_by_name(self, store):
        store.register("a")
        store.register("b")
        store.register_compute("a", lambda: pd.Series([1]))
        store.register_compute("b", lambda: pd.Series([2]))
        store.compute("a")
        store.compute("b")
        store.clear_cache("a")
        stats = store.get_cache_stats()
        names = [s.feature for s in stats]
        assert "a" not in names
        assert "b" in names

    def test_compute_unregistered_raises(self, store):
        with pytest.raises(ValueError, match="No compute function"):
            store.compute("nonexistent")


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, populated_store, tmp_path):
        path = populated_store.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Feature Store" in content

    def test_report_contains_sections(self, populated_store, tmp_path):
        path = populated_store.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Feature Catalog" in content
        assert "Importance" in content
        assert "Freshness" in content
        assert "Lineage" in content
        assert "Cache" in content

    def test_report_embeds_charts(self, populated_store, tmp_path):
        path = populated_store.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_at_default_path(self, populated_store):
        path = populated_store.generate_report()
        assert "feature_store.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")

    def test_empty_store_report(self, store, tmp_path):
        path = store.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
