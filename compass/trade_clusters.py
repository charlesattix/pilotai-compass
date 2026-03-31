"""
Trade clustering analyzer — alias module.

Re-exports everything from :mod:`compass.trade_clustering` under the
``trade_clusters`` name for backward compatibility.

See :mod:`compass.trade_clustering` for full documentation and usage.
"""

from __future__ import annotations

from compass.trade_clustering import (  # noqa: F401
    DEFAULT_CLUSTER_FEATURES,
    ActionableFilter,
    ClusterProfile,
    StabilityResult,
    TradeClusterAnalyzer,
    assess_stability,
    cluster_dbscan,
    cluster_kmeans,
    extract_filters,
    find_optimal_k,
    prepare_cluster_features,
    profile_clusters,
)
