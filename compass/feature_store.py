"""
Feature store manager — centralized feature versioning and lineage tracking.

SQLite-backed storage for feature metadata (name, version, source, dtype,
freshness), with feature registration/retrieval, versioning with lineage
(which features are derived from which), freshness monitoring (stale
feature alerts), feature importance ranking integration, and batch feature
computation with caching.

Generates an HTML report at reports/feature_store.html with feature
catalog, lineage graph, and freshness dashboard.

Usage::

    from compass.feature_store import FeatureStore
    store = FeatureStore("features.db")
    store.register("vix", source="market_data", dtype="float64",
                    description="CBOE VIX index")
    store.set_values("vix", pd.Series([20.5, 21.0, 19.8]))
    store.generate_report("reports/feature_store.html")
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "feature_store.html"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class FeatureMeta:
    """Metadata for a registered feature."""
    name: str
    version: int
    source: str
    dtype: str
    description: str
    created_at: str
    updated_at: str
    row_count: int
    parents: List[str]           # features this one derives from
    importance: Optional[float]  # feature importance score (0-1)
    freshness_seconds: float     # seconds since last update
    tags: List[str]


@dataclass
class FreshnessAlert:
    """Alert for a stale feature."""
    feature: str
    last_updated: str
    stale_seconds: float
    threshold_seconds: float
    severity: str               # "warning" or "critical"


@dataclass
class LineageEdge:
    """Lineage relationship: child derives from parent."""
    parent: str
    child: str
    version: int


@dataclass
class CacheEntry:
    """A cached computation result."""
    feature: str
    cache_key: str
    computed_at: str
    row_count: int
    hit_count: int


# ── SQL schema ──────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    dtype TEXT NOT NULL DEFAULT 'float64',
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    importance REAL,
    tags TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS lineage (
    child TEXT NOT NULL,
    parent TEXT NOT NULL,
    child_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (child, parent, child_version)
);

CREATE TABLE IF NOT EXISTS feature_values (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    idx TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (name, version, idx)
);

CREATE TABLE IF NOT EXISTS cache (
    feature TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    data BLOB,
    hit_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (feature, cache_key)
);

CREATE TABLE IF NOT EXISTS freshness_config (
    feature TEXT PRIMARY KEY,
    threshold_seconds REAL NOT NULL DEFAULT 86400
);
"""


# ── Feature Store ───────────────────────────────────────────────────────


class FeatureStore:
    """SQLite-backed feature store with versioning, lineage, and caching."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._compute_registry: Dict[str, Callable] = {}

    def close(self) -> None:
        self._conn.close()

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        source: str = "",
        dtype: str = "float64",
        description: str = "",
        parents: Optional[List[str]] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        freshness_threshold: float = 86400.0,
    ) -> FeatureMeta:
        """Register a new feature or bump its version."""
        now = datetime.now(timezone.utc).isoformat()
        parents = parents or []
        tags = tags or []

        # Get current max version
        row = self._conn.execute(
            "SELECT MAX(version) FROM features WHERE name = ?", (name,)
        ).fetchone()
        version = (row[0] or 0) + 1

        self._conn.execute(
            """INSERT INTO features (name, version, source, dtype, description,
               created_at, updated_at, row_count, importance, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (name, version, source, dtype, description, now, now,
             importance, json.dumps(tags)),
        )

        # Lineage
        for parent in parents:
            self._conn.execute(
                """INSERT OR REPLACE INTO lineage (child, parent, child_version, created_at)
                   VALUES (?, ?, ?, ?)""",
                (name, parent, version, now),
            )

        # Freshness config
        self._conn.execute(
            """INSERT OR REPLACE INTO freshness_config (feature, threshold_seconds)
               VALUES (?, ?)""",
            (name, freshness_threshold),
        )
        self._conn.commit()

        return self._get_meta(name, version)

    # ── Value storage ───────────────────────────────────────────────────

    def set_values(
        self,
        name: str,
        values: pd.Series,
        version: Optional[int] = None,
    ) -> int:
        """Store feature values. Returns row count."""
        if version is None:
            version = self._latest_version(name)
        if version is None:
            raise ValueError(f"Feature {name!r} not registered")

        now = datetime.now(timezone.utc).isoformat()

        # Clear old values for this version
        self._conn.execute(
            "DELETE FROM feature_values WHERE name = ? AND version = ?",
            (name, version),
        )
        rows = [(name, version, str(idx), float(v) if pd.notna(v) else None)
                for idx, v in values.items()]
        self._conn.executemany(
            "INSERT INTO feature_values (name, version, idx, value) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.execute(
            "UPDATE features SET row_count = ?, updated_at = ? WHERE name = ? AND version = ?",
            (len(rows), now, name, version),
        )
        self._conn.commit()
        return len(rows)

    def get_values(
        self, name: str, version: Optional[int] = None,
    ) -> pd.Series:
        """Retrieve feature values as a pandas Series."""
        if version is None:
            version = self._latest_version(name)
        if version is None:
            raise ValueError(f"Feature {name!r} not registered")

        rows = self._conn.execute(
            "SELECT idx, value FROM feature_values WHERE name = ? AND version = ? ORDER BY idx",
            (name, version),
        ).fetchall()
        if not rows:
            return pd.Series(dtype="float64", name=name)
        idx, vals = zip(*rows)
        return pd.Series(vals, index=list(idx), name=name, dtype="float64")

    # ── Metadata retrieval ──────────────────────────────────────────────

    def get_feature(self, name: str, version: Optional[int] = None) -> FeatureMeta:
        """Get metadata for a feature."""
        if version is None:
            version = self._latest_version(name)
        if version is None:
            raise ValueError(f"Feature {name!r} not registered")
        return self._get_meta(name, version)

    def list_features(self) -> List[FeatureMeta]:
        """List all features (latest version of each)."""
        rows = self._conn.execute(
            """SELECT name, MAX(version) FROM features GROUP BY name ORDER BY name"""
        ).fetchall()
        return [self._get_meta(name, ver) for name, ver in rows]

    def get_lineage(self, name: str) -> List[LineageEdge]:
        """Get all lineage edges for a feature (parents and children)."""
        rows = self._conn.execute(
            """SELECT parent, child, child_version FROM lineage
               WHERE child = ? OR parent = ? ORDER BY child, parent""",
            (name, name),
        ).fetchall()
        return [LineageEdge(parent=r[0], child=r[1], version=r[2]) for r in rows]

    def get_full_lineage(self) -> List[LineageEdge]:
        """Get all lineage edges in the store."""
        rows = self._conn.execute(
            "SELECT parent, child, child_version FROM lineage ORDER BY child, parent"
        ).fetchall()
        return [LineageEdge(parent=r[0], child=r[1], version=r[2]) for r in rows]

    # ── Importance ──────────────────────────────────────────────────────

    def set_importance(self, name: str, importance: float) -> None:
        """Update feature importance score."""
        version = self._latest_version(name)
        if version is None:
            raise ValueError(f"Feature {name!r} not registered")
        self._conn.execute(
            "UPDATE features SET importance = ? WHERE name = ? AND version = ?",
            (importance, name, version),
        )
        self._conn.commit()

    def importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance (descending)."""
        rows = self._conn.execute(
            """SELECT f.name, f.importance FROM features f
               INNER JOIN (SELECT name, MAX(version) as v FROM features GROUP BY name) m
               ON f.name = m.name AND f.version = m.v
               WHERE f.importance IS NOT NULL
               ORDER BY f.importance DESC"""
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ── Freshness monitoring ────────────────────────────────────────────

    def check_freshness(self) -> List[FreshnessAlert]:
        """Check all features for staleness."""
        features = self.list_features()
        alerts: List[FreshnessAlert] = []
        for f in features:
            threshold = self._get_freshness_threshold(f.name)
            if f.freshness_seconds > threshold:
                severity = "critical" if f.freshness_seconds > threshold * 2 else "warning"
                alerts.append(FreshnessAlert(
                    feature=f.name,
                    last_updated=f.updated_at,
                    stale_seconds=f.freshness_seconds,
                    threshold_seconds=threshold,
                    severity=severity,
                ))
        return alerts

    def set_freshness_threshold(self, name: str, seconds: float) -> None:
        """Set freshness threshold for a feature."""
        self._conn.execute(
            "INSERT OR REPLACE INTO freshness_config (feature, threshold_seconds) VALUES (?, ?)",
            (name, seconds),
        )
        self._conn.commit()

    # ── Batch computation with caching ──────────────────────────────────

    def register_compute(self, name: str, fn: Callable[..., pd.Series]) -> None:
        """Register a compute function for batch feature computation."""
        self._compute_registry[name] = fn

    def compute(
        self,
        name: str,
        *args: Any,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> pd.Series:
        """Compute a feature value, using cache if available."""
        if name not in self._compute_registry:
            raise ValueError(f"No compute function registered for {name!r}")

        cache_key = self._make_cache_key(name, args, kwargs)

        if use_cache:
            cached = self._get_cache(name, cache_key)
            if cached is not None:
                return cached

        fn = self._compute_registry[name]
        result = fn(*args, **kwargs)

        # Store in cache
        self._set_cache(name, cache_key, result)

        # Also persist as feature values
        version = self._latest_version(name)
        if version is not None:
            self.set_values(name, result, version)

        return result

    def compute_batch(
        self,
        names: List[str],
        *args: Any,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, pd.Series]:
        """Compute multiple features."""
        return {
            name: self.compute(name, *args, use_cache=use_cache, **kwargs)
            for name in names
            if name in self._compute_registry
        }

    def get_cache_stats(self) -> List[CacheEntry]:
        """Get cache statistics."""
        rows = self._conn.execute(
            "SELECT feature, cache_key, computed_at, row_count, hit_count FROM cache ORDER BY computed_at DESC"
        ).fetchall()
        return [CacheEntry(
            feature=r[0], cache_key=r[1], computed_at=r[2],
            row_count=r[3], hit_count=r[4],
        ) for r in rows]

    def clear_cache(self, name: Optional[str] = None) -> int:
        """Clear cache entries. Returns count deleted."""
        if name:
            cur = self._conn.execute("DELETE FROM cache WHERE feature = ?", (name,))
        else:
            cur = self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        return cur.rowcount

    # ── Internal helpers ────────────────────────────────────────────────

    def _latest_version(self, name: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT MAX(version) FROM features WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def _get_meta(self, name: str, version: int) -> FeatureMeta:
        row = self._conn.execute(
            """SELECT name, version, source, dtype, description,
                      created_at, updated_at, row_count, importance, tags
               FROM features WHERE name = ? AND version = ?""",
            (name, version),
        ).fetchone()
        if row is None:
            raise ValueError(f"Feature {name!r} v{version} not found")

        parents_rows = self._conn.execute(
            "SELECT parent FROM lineage WHERE child = ? AND child_version = ?",
            (name, version),
        ).fetchall()
        parents = [r[0] for r in parents_rows]

        now_ts = datetime.now(timezone.utc)
        try:
            updated = datetime.fromisoformat(row[6])
            freshness = (now_ts - updated).total_seconds()
        except (ValueError, TypeError):
            freshness = 0.0

        return FeatureMeta(
            name=row[0], version=row[1], source=row[2], dtype=row[3],
            description=row[4], created_at=row[5], updated_at=row[6],
            row_count=row[7], parents=parents,
            importance=row[8],
            freshness_seconds=max(freshness, 0.0),
            tags=json.loads(row[9]) if row[9] else [],
        )

    def _get_freshness_threshold(self, name: str) -> float:
        row = self._conn.execute(
            "SELECT threshold_seconds FROM freshness_config WHERE feature = ?",
            (name,),
        ).fetchone()
        return row[0] if row else 86400.0

    @staticmethod
    def _make_cache_key(name: str, args: tuple, kwargs: dict) -> str:
        raw = f"{name}:{repr(args)}:{repr(sorted(kwargs.items()))}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _get_cache(self, name: str, cache_key: str) -> Optional[pd.Series]:
        row = self._conn.execute(
            "SELECT data FROM cache WHERE feature = ? AND cache_key = ?",
            (name, cache_key),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        self._conn.execute(
            "UPDATE cache SET hit_count = hit_count + 1 WHERE feature = ? AND cache_key = ?",
            (name, cache_key),
        )
        self._conn.commit()
        try:
            data = json.loads(row[0])
            return pd.Series(data.get("values", []),
                             index=data.get("index", []),
                             name=name, dtype="float64")
        except (json.JSONDecodeError, TypeError):
            return None

    def _set_cache(self, name: str, cache_key: str, result: pd.Series) -> None:
        now = datetime.now(timezone.utc).isoformat()
        data = json.dumps({"index": list(result.index.astype(str)),
                           "values": [float(v) if pd.notna(v) else None for v in result.values]})
        self._conn.execute(
            """INSERT OR REPLACE INTO cache (feature, cache_key, computed_at, row_count, data, hit_count)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (name, cache_key, now, len(result), data),
        )
        self._conn.commit()

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report with feature catalog, lineage, freshness."""
        features = self.list_features()
        lineage = self.get_full_lineage()
        alerts = self.check_freshness()
        ranking = self.importance_ranking()
        cache_stats = self.get_cache_stats()

        charts = self._render_charts(features, lineage, ranking)
        html = self._build_html(features, lineage, alerts, ranking, cache_stats, charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    # ── Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(
        self,
        features: List[FeatureMeta],
        lineage: List[LineageEdge],
        ranking: List[Tuple[str, float]],
    ) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["importance"] = self._chart_importance(ranking)
        charts["freshness"] = self._chart_freshness(features)
        charts["lineage"] = self._chart_lineage(features, lineage)
        return charts

    def _chart_importance(self, ranking: List[Tuple[str, float]]) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not ranking:
            return ""
        names = [r[0] for r in ranking[:20]]
        vals = [r[1] for r in ranking[:20]]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.35)))
        colors = ["#3b82f6" if v >= np.median(vals) else "#94a3b8" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance Ranking", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_freshness(self, features: List[FeatureMeta]) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not features:
            return ""
        fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.35)))
        names = [f.name for f in features]
        hours = [f.freshness_seconds / 3600 for f in features]
        thresholds = [self._get_freshness_threshold(f.name) / 3600 for f in features]
        colors = ["#dc2626" if h > t else "#16a34a" for h, t in zip(hours, thresholds)]
        ax.barh(names, hours, color=colors, alpha=0.85)
        for i, t in enumerate(thresholds):
            ax.plot(t, i, "k|", markersize=12, markeredgewidth=2)
        ax.set_xlabel("Hours Since Update")
        ax.set_title("Feature Freshness Dashboard", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_lineage(
        self, features: List[FeatureMeta], lineage: List[LineageEdge],
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not features:
            return ""
        all_names = {f.name for f in features}
        for e in lineage:
            all_names.add(e.parent)
            all_names.add(e.child)
        names = sorted(all_names)
        if not names:
            return ""

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), max(4, len(names) * 0.5)))

        # Layout: simple layered positioning
        name_to_idx = {n: i for i, n in enumerate(names)}
        # Determine depth: features with no parents are depth 0
        depths: Dict[str, int] = {}
        for n in names:
            parents = [e.parent for e in lineage if e.child == n]
            depths[n] = 0 if not parents else 1
        # Second pass for deeper chains
        for _ in range(3):
            for e in lineage:
                if e.parent in depths and e.child in depths:
                    depths[e.child] = max(depths[e.child], depths[e.parent] + 1)

        max_depth = max(depths.values()) if depths else 0
        by_depth: Dict[int, List[str]] = {}
        for n, d in depths.items():
            by_depth.setdefault(d, []).append(n)

        positions: Dict[str, Tuple[float, float]] = {}
        for d, ns in by_depth.items():
            for i, n in enumerate(ns):
                x = d * 2.0
                y = i - len(ns) / 2.0
                positions[n] = (x, y)

        # Draw edges
        for e in lineage:
            if e.parent in positions and e.child in positions:
                px, py = positions[e.parent]
                cx, cy = positions[e.child]
                ax.annotate("", xy=(cx, cy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

        # Draw nodes
        for n, (x, y) in positions.items():
            has_importance = any(f.importance is not None and f.importance > 0
                                for f in features if f.name == n)
            color = "#3b82f6" if has_importance else "#94a3b8"
            ax.scatter(x, y, s=200, color=color, zorder=5, edgecolors="white", linewidth=1.5)
            ax.annotate(n, (x, y), fontsize=7, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

        ax.set_title("Feature Lineage Graph", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(
        self,
        features: List[FeatureMeta],
        lineage: List[LineageEdge],
        alerts: List[FreshnessAlert],
        ranking: List[Tuple[str, float]],
        cache_stats: List[CacheEntry],
        charts: Dict[str, str],
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Feature catalog table
        catalog_rows = ""
        for f in features:
            imp_str = f"{f.importance:.3f}" if f.importance is not None else "—"
            parents_str = ", ".join(f.parents) if f.parents else "—"
            tags_str = ", ".join(f.tags) if f.tags else "—"
            catalog_rows += (
                f'<tr><td>{f.name}</td><td>v{f.version}</td><td>{f.source}</td>'
                f'<td>{f.dtype}</td><td>{f.row_count:,}</td>'
                f'<td>{imp_str}</td><td>{parents_str}</td>'
                f'<td>{tags_str}</td></tr>\n'
            )

        # Freshness alerts
        alert_rows = ""
        for a in alerts:
            cls = "bad" if a.severity == "critical" else "warn"
            alert_rows += (
                f'<tr><td>{a.feature}</td>'
                f'<td>{a.stale_seconds / 3600:.1f}h</td>'
                f'<td>{a.threshold_seconds / 3600:.1f}h</td>'
                f'<td class="{cls}">{a.severity.upper()}</td></tr>\n'
            )
        if not alert_rows:
            alert_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">All features fresh</td></tr>'

        # Lineage table
        lineage_rows = ""
        for e in lineage:
            lineage_rows += f'<tr><td>{e.parent}</td><td>{e.child}</td><td>v{e.version}</td></tr>\n'
        if not lineage_rows:
            lineage_rows = '<tr><td colspan="3" style="text-align:center;color:#64748b">No lineage edges</td></tr>'

        # Cache stats
        cache_rows = ""
        for c in cache_stats[:20]:
            cache_rows += (
                f'<tr><td>{c.feature}</td><td>{c.cache_key}</td>'
                f'<td>{c.row_count}</td><td>{c.hit_count}</td></tr>\n'
            )
        if not cache_rows:
            cache_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">Cache empty</td></tr>'

        n_stale = len(alerts)
        stale_cls = "bad" if n_stale > 0 else "good"
        total_rows = sum(f.row_count for f in features)

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Feature Store Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .warn {{ color: #f59e0b; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Feature Store Dashboard</h1>
<div class="meta">{len(features)} features &middot; {total_rows:,} total rows &middot; {len(lineage)} lineage edges &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{len(features)}</div><div class="label">Features</div></div>
  <div class="kpi"><div class="value">{total_rows:,}</div><div class="label">Total Rows</div></div>
  <div class="kpi"><div class="value">{len(lineage)}</div><div class="label">Lineage Edges</div></div>
  <div class="kpi"><div class="value {stale_cls}">{n_stale}</div><div class="label">Stale Alerts</div></div>
  <div class="kpi"><div class="value">{len(cache_stats)}</div><div class="label">Cache Entries</div></div>
</div>

<h2>1. Feature Catalog</h2>
<table>
<thead><tr><th>Name</th><th>Version</th><th>Source</th><th>Dtype</th><th>Rows</th><th>Importance</th><th>Parents</th><th>Tags</th></tr></thead>
<tbody>{catalog_rows}</tbody>
</table>

<h2>2. Feature Importance</h2>
{_img("importance")}

<h2>3. Freshness Dashboard</h2>
{_img("freshness")}
<table>
<thead><tr><th>Feature</th><th>Age</th><th>Threshold</th><th>Status</th></tr></thead>
<tbody>{alert_rows}</tbody>
</table>

<h2>4. Lineage Graph</h2>
{_img("lineage")}
<table>
<thead><tr><th>Parent</th><th>Child</th><th>Child Version</th></tr></thead>
<tbody>{lineage_rows}</tbody>
</table>

<h2>5. Cache Statistics</h2>
<table>
<thead><tr><th>Feature</th><th>Cache Key</th><th>Rows</th><th>Hits</th></tr></thead>
<tbody>{cache_rows}</tbody>
</table>

<footer>Generated by <code>compass/feature_store.py</code></footer>
</body></html>"""
        return html
