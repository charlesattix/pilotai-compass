"""
Trade clustering analyzer — unsupervised discovery of trade archetypes.

Uses KMeans and DBSCAN to find natural clusters in trade entry features,
then profiles each cluster by win rate, avg P&L, and risk to identify
actionable filters (feature ranges that produce the best outcomes).

Key outputs:
  1. Cluster assignments with profiling (win rate, avg P&L per cluster)
  2. Cluster centroids in feature space → human-readable profiles
  3. Actionable filters: feature-range rules that isolate high-Sharpe clusters
  4. Walk-forward cluster stability (do the same clusters persist?)
  5. HTML report with PCA scatter plot, profile cards, filter recommendations

Usage::

    from compass.trade_clustering import TradeClusterAnalyzer
    analyzer = TradeClusterAnalyzer.from_csv("compass/training_data_combined.csv")
    results = analyzer.fit()
    analyzer.generate_report("reports/trade_clusters.html")
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "trade_clusters.html"

# Features used for clustering (numeric, non-target, non-leaky)
DEFAULT_CLUSTER_FEATURES = [
    "dte_at_entry", "hold_days", "rsi_14",
    "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_50d", "iv_rank",
    "dist_from_ma20_pct", "dist_from_ma50_pct",
    "dist_from_ma200_pct", "ma50_slope_ann_pct",
    "realized_vol_20d", "net_credit",
    "spread_width", "max_loss_per_unit", "contracts",
]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ClusterProfile:
    """Profile of a single trade cluster."""
    cluster_id: int
    n_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    avg_return_pct: float
    sharpe: Optional[float]
    dominant_regime: str
    dominant_exit: str
    centroid: Dict[str, float]          # feature name → centroid value
    feature_ranges: Dict[str, Tuple[float, float]]  # feature → (p25, p75)
    label: str                          # human-readable: "High Win", "Risky", etc.


@dataclass
class ActionableFilter:
    """A feature-range rule that isolates a high-quality cluster."""
    feature: str
    low: float
    high: float
    cluster_id: int
    win_rate: float
    avg_pnl: float
    n_trades: int
    description: str


@dataclass
class StabilityResult:
    """Walk-forward cluster stability analysis."""
    n_folds: int
    avg_ari: float          # Adjusted Rand Index across folds
    avg_silhouette: float
    stable: bool            # True if clusters are consistent


# ── Core clustering ──────────────────────────────────────────────────────


def prepare_cluster_features(
    trades: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Scale and impute features for clustering.

    Returns (X_scaled, used_columns, fitted_scaler).
    """
    cols = feature_cols or DEFAULT_CLUSTER_FEATURES
    available = [c for c in cols if c in trades.columns]
    X = trades[available].copy().fillna(0).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, available, scaler


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8)) -> int:
    """Find optimal k for KMeans using silhouette score."""
    best_k = 2
    best_score = -1.0
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_kmeans(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run KMeans clustering. Returns (labels, centroids)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def cluster_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 5) -> np.ndarray:
    """Run DBSCAN clustering. Returns labels (-1 = noise)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


# ── Cluster profiling ────────────────────────────────────────────────────


def profile_clusters(
    trades: pd.DataFrame,
    labels: np.ndarray,
    centroids: Optional[np.ndarray],
    feature_cols: List[str],
) -> List[ClusterProfile]:
    """Build a ClusterProfile for each cluster."""
    profiles = []
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)  # DBSCAN noise

    for cid in unique_labels:
        mask = labels == cid
        cluster = trades[mask]
        n = len(cluster)
        if n == 0:
            continue

        pnls = cluster["pnl"].dropna()
        wins = cluster["win"].dropna() if "win" in cluster.columns else pd.Series(dtype=float)

        win_rate = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_pnl = float(pnls.mean()) if len(pnls) > 0 else 0.0
        total_pnl = float(pnls.sum()) if len(pnls) > 0 else 0.0
        avg_ret = float(cluster["return_pct"].mean()) if "return_pct" in cluster.columns else 0.0

        sharpe = None
        if len(pnls) > 1:
            std = float(pnls.std(ddof=1))
            if std > 0:
                sharpe = round(avg_pnl / std * math.sqrt(52), 3)

        regime_counts = cluster["regime"].value_counts() if "regime" in cluster.columns else pd.Series(dtype=int)
        dominant_regime = str(regime_counts.index[0]) if len(regime_counts) > 0 else "?"

        exit_counts = cluster["exit_reason"].value_counts() if "exit_reason" in cluster.columns else pd.Series(dtype=int)
        dominant_exit = str(exit_counts.index[0]) if len(exit_counts) > 0 else "?"

        # Centroid
        centroid_dict = {}
        if centroids is not None and cid < len(centroids):
            for i, col in enumerate(feature_cols):
                if i < centroids.shape[1]:
                    centroid_dict[col] = round(float(centroids[cid, i]), 4)

        # Feature ranges (p25, p75)
        ranges = {}
        for col in feature_cols:
            if col in cluster.columns:
                vals = cluster[col].dropna()
                if len(vals) > 0:
                    ranges[col] = (round(float(vals.quantile(0.25)), 4),
                                   round(float(vals.quantile(0.75)), 4))

        # Label
        if win_rate >= 0.65 and avg_pnl > 0:
            label = "High Win"
        elif win_rate < 0.45:
            label = "Risky"
        elif avg_pnl > 0:
            label = "Moderate"
        else:
            label = "Underperforming"

        profiles.append(ClusterProfile(
            cluster_id=cid, n_trades=n, win_rate=round(win_rate, 4),
            avg_pnl=round(avg_pnl, 2), total_pnl=round(total_pnl, 2),
            avg_return_pct=round(avg_ret, 2),
            sharpe=sharpe, dominant_regime=dominant_regime,
            dominant_exit=dominant_exit, centroid=centroid_dict,
            feature_ranges=ranges, label=label,
        ))

    return profiles


# ── Actionable filters ───────────────────────────────────────────────────


def extract_filters(
    profiles: List[ClusterProfile],
    min_trades: int = 20,
    min_win_rate: float = 0.60,
) -> List[ActionableFilter]:
    """Extract feature-range filters from high-performing clusters."""
    filters = []
    good_clusters = [p for p in profiles if p.win_rate >= min_win_rate
                     and p.n_trades >= min_trades and p.avg_pnl > 0]

    for cluster in good_clusters:
        for feat, (lo, hi) in cluster.feature_ranges.items():
            if lo == hi:
                continue
            filters.append(ActionableFilter(
                feature=feat, low=lo, high=hi,
                cluster_id=cluster.cluster_id,
                win_rate=cluster.win_rate,
                avg_pnl=cluster.avg_pnl,
                n_trades=cluster.n_trades,
                description=f"Cluster {cluster.cluster_id} ({cluster.label}): "
                            f"{feat} in [{lo:.2f}, {hi:.2f}]",
            ))

    # Rank by win_rate × n_trades (prefer filters with more evidence)
    filters.sort(key=lambda f: -(f.win_rate * f.n_trades))
    return filters


# ── Cluster stability ────────────────────────────────────────────────────


def assess_stability(
    trades: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    n_folds: int = 3,
) -> StabilityResult:
    """Walk-forward cluster stability via Adjusted Rand Index."""
    from sklearn.metrics import adjusted_rand_score

    n = len(trades)
    fold_size = n // (n_folds + 1)
    if fold_size < 20:
        return StabilityResult(0, 0.0, 0.0, False)

    ari_scores = []
    sil_scores = []

    prev_labels = None
    for fold in range(n_folds):
        end = fold_size * (fold + 2)
        subset = trades.iloc[:end]
        X, cols, _ = prepare_cluster_features(subset, feature_cols)
        if len(X) < k + 1:
            continue
        labels, _ = cluster_kmeans(X, k)

        if len(set(labels)) >= 2:
            sil_scores.append(silhouette_score(X, labels))

        if prev_labels is not None:
            overlap = min(len(prev_labels), len(labels))
            if overlap > 10:
                ari = adjusted_rand_score(prev_labels[:overlap], labels[:overlap])
                ari_scores.append(ari)
        prev_labels = labels

    avg_ari = float(np.mean(ari_scores)) if ari_scores else 0.0
    avg_sil = float(np.mean(sil_scores)) if sil_scores else 0.0

    return StabilityResult(
        n_folds=len(ari_scores),
        avg_ari=round(avg_ari, 4),
        avg_silhouette=round(avg_sil, 4),
        stable=avg_ari > 0.3,
    )


# ── TradeClusterAnalyzer ────────────────────────────────────────────────


class TradeClusterAnalyzer:
    """Unsupervised trade archetype discovery and profiling.

    Args:
        trades: DataFrame of closed trades.
        feature_cols: Features to cluster on (default: DEFAULT_CLUSTER_FEATURES).
        max_k: Maximum number of clusters to try.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        max_k: int = 7,
    ):
        self.trades = trades.copy()
        self.feature_cols = feature_cols or DEFAULT_CLUSTER_FEATURES
        self.max_k = max_k

        self.X_scaled: Optional[np.ndarray] = None
        self.used_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.labels: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.optimal_k: int = 3
        self.profiles: List[ClusterProfile] = []
        self.filters: List[ActionableFilter] = []
        self.stability: Optional[StabilityResult] = None
        self.silhouette: Optional[float] = None
        self.pca_2d: Optional[np.ndarray] = None

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "TradeClusterAnalyzer":
        return cls(pd.read_csv(csv_path), **kwargs)

    def fit(self) -> Dict[str, Any]:
        """Run full clustering pipeline."""
        # 1. Prepare features
        self.X_scaled, self.used_cols, self.scaler = prepare_cluster_features(
            self.trades, self.feature_cols,
        )

        # 2. Find optimal k
        self.optimal_k = find_optimal_k(self.X_scaled, range(2, self.max_k + 1))
        logger.info("Optimal k = %d", self.optimal_k)

        # 3. Cluster
        self.labels, self.centroids = cluster_kmeans(self.X_scaled, self.optimal_k)
        self.trades["cluster"] = self.labels

        # 4. Silhouette
        if len(set(self.labels)) >= 2:
            self.silhouette = round(silhouette_score(self.X_scaled, self.labels), 4)

        # 5. PCA for visualization
        if self.X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            self.pca_2d = pca.fit_transform(self.X_scaled)

        # 6. Profile clusters
        self.profiles = profile_clusters(
            self.trades, self.labels, self.centroids, self.used_cols,
        )

        # 7. Extract actionable filters
        self.filters = extract_filters(self.profiles)

        # 8. Stability analysis
        self.stability = assess_stability(
            self.trades, self.used_cols, self.optimal_k,
        )

        return {
            "optimal_k": self.optimal_k,
            "silhouette": self.silhouette,
            "profiles": self.profiles,
            "filters": self.filters,
            "stability": self.stability,
        }

    # ── HTML Report ──────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if not self.profiles:
            self.fit()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        if self.pca_2d is not None:
            charts["scatter"] = self._chart_scatter()
        charts["profile_bars"] = self._chart_profile_bars()
        return charts

    def _chart_scatter(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(self.optimal_k, 1)))
        wins = self.trades["win"].values if "win" in self.trades.columns else np.zeros(len(self.trades))

        for cid in range(self.optimal_k):
            mask = self.labels == cid
            profile = next((p for p in self.profiles if p.cluster_id == cid), None)
            lbl = f"C{cid}: {profile.label} (WR={profile.win_rate:.0%})" if profile else f"C{cid}"
            ax.scatter(self.pca_2d[mask, 0], self.pca_2d[mask, 1],
                       c=[colors[cid]], alpha=0.6, s=30, label=lbl,
                       edgecolors="white", linewidths=0.3)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Trade Clusters (k={self.optimal_k}, silhouette={self.silhouette or 0:.3f})", fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_profile_bars(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.profiles:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        names = [f"C{p.cluster_id}\n({p.label})" for p in self.profiles]
        win_rates = [p.win_rate for p in self.profiles]
        avg_pnls = [p.avg_pnl for p in self.profiles]

        wr_colors = ["#16a34a" if wr >= 0.55 else "#d97706" if wr >= 0.45 else "#dc2626" for wr in win_rates]
        pnl_colors = ["#16a34a" if p >= 0 else "#dc2626" for p in avg_pnls]

        ax1.bar(range(len(names)), [wr * 100 for wr in win_rates], color=wr_colors, alpha=0.85)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, fontsize=8)
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_title("Win Rate by Cluster", fontsize=10)
        ax1.axhline(50, color="gray", ls="--", lw=0.8)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(range(len(names)), avg_pnls, color=pnl_colors, alpha=0.85)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, fontsize=8)
        ax2.set_ylabel("Avg P&L ($)")
        ax2.set_title("Avg P&L by Cluster", fontsize=10)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ''

        # Profile cards
        profile_cards = ""
        for p in sorted(self.profiles, key=lambda x: -x.win_rate):
            pnl_cls = "good" if p.avg_pnl >= 0 else "bad"
            top_feats = sorted(p.feature_ranges.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True)[:5]
            feat_rows = "".join(
                f'<tr><td>{f}</td><td>[{lo:.2f}, {hi:.2f}]</td></tr>'
                for f, (lo, hi) in top_feats
            )
            profile_cards += f"""
            <div class="profile-card {'good-border' if p.label == 'High Win' else 'bad-border' if p.label == 'Risky' else ''}">
              <h3>Cluster {p.cluster_id}: {p.label}</h3>
              <div class="card-kpis">
                <span>{p.n_trades} trades</span> &middot;
                <span>WR: {p.win_rate:.0%}</span> &middot;
                <span class="{pnl_cls}">Avg P&L: ${p.avg_pnl:,.0f}</span> &middot;
                <span>Sharpe: {p.sharpe or '—'}</span>
              </div>
              <p>Regime: <strong>{p.dominant_regime}</strong> | Exit: {p.dominant_exit}</p>
              <table class="feat-table"><thead><tr><th>Feature</th><th>IQR Range</th></tr></thead>
              <tbody>{feat_rows}</tbody></table>
            </div>"""

        # Filters table
        filter_rows = ""
        for f in self.filters[:15]:
            filter_rows += (
                f'<tr><td>{f.feature}</td><td>[{f.low:.2f}, {f.high:.2f}]</td>'
                f'<td>C{f.cluster_id}</td><td>{f.win_rate:.0%}</td>'
                f'<td>${f.avg_pnl:,.0f}</td><td>{f.n_trades}</td></tr>\n'
            )
        if not filter_rows:
            filter_rows = '<tr><td colspan="6" class="muted">No high-confidence filters found</td></tr>'

        stab = self.stability
        stab_html = ""
        if stab:
            badge = "stable" if stab.stable else "unstable"
            stab_html = (
                f'<p>Walk-forward stability: <span class="badge {badge}">'
                f'{"STABLE" if stab.stable else "UNSTABLE"}</span> '
                f'(ARI={stab.avg_ari:.3f}, silhouette={stab.avg_silhouette:.3f}, '
                f'{stab.n_folds} folds)</p>'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trade Cluster Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  h3 {{ margin: 0.5em 0 0.3em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .muted {{ color: #94a3b8; font-style: italic; text-align: center; padding: 1em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .profile-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                   padding: 1.2em; margin: 1em 0; }}
  .good-border {{ border-left: 4px solid #16a34a; }}
  .bad-border {{ border-left: 4px solid #dc2626; }}
  .card-kpis {{ font-size: 0.9em; color: #475569; margin: 0.3em 0; }}
  .feat-table {{ max-width: 400px; font-size: 0.82em; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.78em; font-weight: 600; }}
  .badge.stable {{ background: #dcfce7; color: #166534; }}
  .badge.unstable {{ background: #fef3c7; color: #92400e; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Trade Cluster Analysis</h1>
<div class="meta">{len(self.trades)} trades &middot; {self.optimal_k} clusters &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{self.optimal_k}</div><div class="label">Clusters</div></div>
  <div class="kpi"><div class="value">{self.silhouette or '—'}</div><div class="label">Silhouette</div></div>
  <div class="kpi"><div class="value">{len(self.used_cols)}</div><div class="label">Features</div></div>
  <div class="kpi"><div class="value">{len(self.filters)}</div><div class="label">Actionable Filters</div></div>
</div>

<h2>1. Cluster Scatter (PCA)</h2>
{_img("scatter")}

<h2>2. Cluster Profiles</h2>
{_img("profile_bars")}
{stab_html}
{profile_cards}

<h2>3. Actionable Filters</h2>
<p>Feature ranges from high-performing clusters (win rate &ge; 60%, positive avg P&L).</p>
<table>
<thead><tr><th>Feature</th><th>Range</th><th>Cluster</th><th>Win Rate</th><th>Avg P&L</th><th>Trades</th></tr></thead>
<tbody>{filter_rows}</tbody>
</table>

<footer>Generated by <code>compass/trade_clustering.py</code></footer>
</body></html>"""
        return html
