"""
Cross-strategy correlation and diversification analysis.

Pairwise return correlations (rolling + full-sample), conditional
correlations (crisis vs calm), diversification ratio, marginal risk
contribution, optimal max-diversification weights, strategy clustering
(hierarchical + KMeans), regime-dependent correlation shifts, tail
dependence estimation, and portfolio recommendations.

Usage::

    from compass.strategy_correlation import StrategyCorrelationAnalyzer
    analyzer = StrategyCorrelationAnalyzer(strategy_returns)
    results = analyzer.analyze()
    analyzer.generate_report()
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "strategy_correlation.html"

REGIMES = ("bull", "bear", "high_vol", "neutral")
CRISIS_REGIMES = ("bear", "high_vol")
CALM_REGIMES = ("bull", "neutral")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class PairCorrelation:
    """Pairwise correlation between two strategies."""
    strategy_a: str
    strategy_b: str
    full_corr: float
    rolling_mean: float
    rolling_std: float
    current_corr: float
    crisis_corr: float
    calm_corr: float
    correlation_shift: float  # crisis - calm
    tail_dependence: float    # lower tail dependence


@dataclass
class DiversificationMetrics:
    """Portfolio-level diversification metrics."""
    diversification_ratio: float   # weighted avg vol / portfolio vol
    effective_n: float             # 1 / HHI of risk contributions
    avg_correlation: float
    max_correlation: float
    min_correlation: float


@dataclass
class MarginalRisk:
    """Marginal risk contribution for one strategy."""
    strategy: str
    weight: float
    marginal_contribution: float   # % of portfolio risk
    standalone_vol: float
    beta_to_portfolio: float


@dataclass
class ClusterResult:
    """Strategy clustering result."""
    cluster_id: int
    strategies: List[str]
    avg_intra_corr: float
    representative: str       # most central strategy


@dataclass
class RegimeCorrelationShift:
    """How correlations change across regimes."""
    regime: str
    avg_corr: float
    max_corr: float
    n_obs: int
    shift_from_overall: float


@dataclass
class Recommendation:
    """Portfolio recommendation."""
    action: str               # "add", "remove", "reweight"
    strategy: str
    reason: str
    estimated_impact: float   # expected diversification improvement
    priority: str             # "high", "medium", "low"


@dataclass
class OptimalWeights:
    """Max-diversification optimal weights."""
    weights: Dict[str, float]
    diversification_ratio: float
    portfolio_vol: float
    sharpe: float


# ── Analyzer ────────────────────────────────────────────────────────────


class StrategyCorrelationAnalyzer:
    """Cross-strategy correlation and diversification analysis."""

    def __init__(
        self,
        returns: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        window: int = 60,
        weights: Optional[Dict[str, float]] = None,
        n_clusters: int = 3,
    ) -> None:
        self.returns = returns.dropna(how="all").copy()
        self.strategies = list(returns.columns)
        self.regimes = regimes if regimes is not None else pd.Series(
            "neutral", index=returns.index,
        )
        self.window = window
        n = len(self.strategies)
        self.weights = weights or {s: 1.0 / n for s in self.strategies}
        self.n_clusters = min(n_clusters, n)

        # Align
        common = self.returns.index.intersection(self.regimes.index)
        self.returns = self.returns.loc[common]
        self.regimes = self.regimes.loc[common]

        # Results
        self.pair_correlations: List[PairCorrelation] = []
        self.diversification: Optional[DiversificationMetrics] = None
        self.marginal_risks: List[MarginalRisk] = []
        self.clusters: List[ClusterResult] = []
        self.regime_shifts: List[RegimeCorrelationShift] = []
        self.recommendations: List[Recommendation] = []
        self.optimal_weights: Optional[OptimalWeights] = None

    @classmethod
    def from_csv(
        cls, returns_path: str, regimes_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "StrategyCorrelationAnalyzer":
        ret = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        reg = None
        if regimes_path:
            reg = pd.read_csv(regimes_path, index_col=0, parse_dates=True).iloc[:, 0]
        return cls(ret, regimes=reg, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.pair_correlations = self._pairwise_correlations()
        self.diversification = self._diversification_metrics()
        self.marginal_risks = self._marginal_risk_contributions()
        self.clusters = self._cluster_strategies()
        self.regime_shifts = self._regime_correlation_shifts()
        self.optimal_weights = self._max_diversification_weights()
        self.recommendations = self._generate_recommendations()
        return {
            "pair_correlations": self.pair_correlations,
            "diversification": self.diversification,
            "marginal_risks": self.marginal_risks,
            "clusters": self.clusters,
            "regime_shifts": self.regime_shifts,
            "optimal_weights": self.optimal_weights,
            "recommendations": self.recommendations,
        }

    # ── Pairwise correlations ───────────────────────────────────────────

    def _pairwise_correlations(self) -> List[PairCorrelation]:
        results: List[PairCorrelation] = []
        n = len(self.strategies)
        crisis_mask = self.regimes.isin(CRISIS_REGIMES)
        calm_mask = self.regimes.isin(CALM_REGIMES)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.strategies[i], self.strategies[j]
                ra, rb = self.returns[a], self.returns[b]

                full = float(ra.corr(rb))

                # Rolling
                rolling = ra.rolling(self.window).corr(rb).dropna()
                r_mean = float(rolling.mean()) if len(rolling) > 0 else full
                r_std = float(rolling.std()) if len(rolling) > 1 else 0
                current = float(rolling.iloc[-1]) if len(rolling) > 0 else full

                # Conditional
                crisis_sub_a = ra.loc[crisis_mask].dropna()
                crisis_sub_b = rb.loc[crisis_mask].dropna()
                common_crisis = crisis_sub_a.index.intersection(crisis_sub_b.index)
                crisis_corr = float(crisis_sub_a.loc[common_crisis].corr(
                    crisis_sub_b.loc[common_crisis])) if len(common_crisis) > 5 else full

                calm_sub_a = ra.loc[calm_mask].dropna()
                calm_sub_b = rb.loc[calm_mask].dropna()
                common_calm = calm_sub_a.index.intersection(calm_sub_b.index)
                calm_corr = float(calm_sub_a.loc[common_calm].corr(
                    calm_sub_b.loc[common_calm])) if len(common_calm) > 5 else full

                shift = crisis_corr - calm_corr

                # Tail dependence: empirical lower tail
                tail_dep = self._tail_dependence(ra.values, rb.values)

                results.append(PairCorrelation(
                    a, b, full, r_mean, r_std, current,
                    crisis_corr, calm_corr, shift, tail_dep,
                ))
        return sorted(results, key=lambda p: -abs(p.full_corr))

    @staticmethod
    def _tail_dependence(x: np.ndarray, y: np.ndarray, q: float = 0.05) -> float:
        """Empirical lower tail dependence coefficient."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 20:
            return 0.0
        # Rank transform to uniform
        from scipy.stats import rankdata
        u = rankdata(x) / (len(x) + 1)
        v = rankdata(y) / (len(y) + 1)
        # P(V <= q | U <= q) — empirical
        below_u = u <= q
        if below_u.sum() == 0:
            return 0.0
        return float((v[below_u] <= q).mean())

    # ── Diversification metrics ─────────────────────────────────────────

    def _diversification_metrics(self) -> DiversificationMetrics:
        cov = self.returns.cov().values
        w = np.array([self.weights[s] for s in self.strategies])
        w = w / w.sum()

        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 0))
        individual_vols = np.sqrt(np.diag(cov))
        weighted_avg_vol = float(w @ individual_vols)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0

        # Effective N
        if port_vol > 0:
            mc = (cov @ w) / port_vol
            risk_contribs = w * mc
            rc_pct = risk_contribs / risk_contribs.sum() if risk_contribs.sum() > 0 else w
            hhi = float(np.sum(rc_pct ** 2))
            eff_n = 1.0 / hhi if hhi > 0 else len(self.strategies)
        else:
            eff_n = float(len(self.strategies))

        corr = self.returns.corr().values.copy()
        np.fill_diagonal(corr, np.nan)
        avg_c = float(np.nanmean(corr))
        max_c = float(np.nanmax(corr))
        min_c = float(np.nanmin(corr))

        return DiversificationMetrics(div_ratio, eff_n, avg_c, max_c, min_c)

    # ── Marginal risk ───────────────────────────────────────────────────

    def _marginal_risk_contributions(self) -> List[MarginalRisk]:
        cov = self.returns.cov().values
        w = np.array([self.weights[s] for s in self.strategies])
        w = w / w.sum()
        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 0))
        individual_vols = np.sqrt(np.diag(cov))

        results: List[MarginalRisk] = []
        for i, s in enumerate(self.strategies):
            if port_vol > 0:
                mc = float(w[i] * (cov[i] @ w) / port_vol)
            else:
                mc = 0.0
            beta = float((cov[i] @ w) / port_var) if port_var > 0 else 0.0
            results.append(MarginalRisk(
                s, float(w[i]), mc, float(individual_vols[i]), beta,
            ))
        return sorted(results, key=lambda m: -m.marginal_contribution)

    # ── Clustering ──────────────────────────────────────────────────────

    def _cluster_strategies(self) -> List[ClusterResult]:
        n = len(self.strategies)
        if n < 2:
            return [ClusterResult(0, self.strategies, 0, self.strategies[0] if self.strategies else "")]

        corr = self.returns.corr().values.copy()
        # Distance = 1 - corr (ensure non-negative)
        dist = np.clip(1 - corr, 0, 2)
        np.fill_diagonal(dist, 0)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=self.n_clusters, criterion="maxclust")

        clusters: Dict[int, List[str]] = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(self.strategies[i])

        results: List[ClusterResult] = []
        for cid, members in sorted(clusters.items()):
            # Intra-cluster correlation
            if len(members) >= 2:
                sub = self.returns[members].corr().values.copy()
                np.fill_diagonal(sub, np.nan)
                avg_intra = float(np.nanmean(sub))
            else:
                avg_intra = 1.0

            # Representative: highest avg corr with others in cluster
            if len(members) >= 2:
                best_s = max(members, key=lambda m: float(
                    self.returns[members].corr()[m].drop(m).mean()
                ))
            else:
                best_s = members[0]

            results.append(ClusterResult(cid, members, avg_intra, best_s))
        return results

    # ── Regime shifts ───────────────────────────────────────────────────

    def _regime_correlation_shifts(self) -> List[RegimeCorrelationShift]:
        overall_corr = self.returns.corr().values.copy()
        np.fill_diagonal(overall_corr, np.nan)
        overall_avg = float(np.nanmean(overall_corr))

        results: List[RegimeCorrelationShift] = []
        for regime in REGIMES:
            mask = self.regimes == regime
            sub = self.returns.loc[mask]
            if len(sub) < 5:
                continue
            corr = sub.corr().values.copy()
            np.fill_diagonal(corr, np.nan)
            avg = float(np.nanmean(corr))
            mx = float(np.nanmax(corr))
            results.append(RegimeCorrelationShift(
                regime, avg, mx, int(mask.sum()), avg - overall_avg,
            ))
        return results

    # ── Max diversification weights ─────────────────────────────────────

    def _max_diversification_weights(self) -> OptimalWeights:
        """Maximize diversification ratio = weighted_avg_vol / portfolio_vol."""
        cov = self.returns.cov().values
        n = len(self.strategies)
        vols = np.sqrt(np.diag(cov))

        # Analytical: w* ∝ Σ^{-1} σ  (inverse-variance weighted by vol)
        try:
            reg = np.eye(n) * 1e-8
            inv_cov = np.linalg.inv(cov + reg)
            w = inv_cov @ vols
            w = np.maximum(w, 0)  # long-only
            w_sum = w.sum()
            w = w / w_sum if w_sum > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            w = np.ones(n) / n

        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 0))
        wavg_vol = float(w @ vols)
        div_ratio = wavg_vol / port_vol if port_vol > 0 else 1.0

        # Sharpe estimate
        mu = self.returns.mean().values
        port_ret = float(w @ mu) * 252
        sh = port_ret / (port_vol * math.sqrt(252)) if port_vol > 0 else 0

        return OptimalWeights(
            weights={s: float(w[i]) for i, s in enumerate(self.strategies)},
            diversification_ratio=div_ratio,
            portfolio_vol=port_vol,
            sharpe=sh,
        )

    # ── Recommendations ─────────────────────────────────────────────────

    def _generate_recommendations(self) -> List[Recommendation]:
        recs: List[Recommendation] = []

        # Redundant: strategies in same cluster with very high correlation
        for cl in self.clusters:
            if cl.avg_intra_corr > 0.7 and len(cl.strategies) > 1:
                for s in cl.strategies:
                    if s != cl.representative:
                        recs.append(Recommendation(
                            "remove", s,
                            f"Redundant with {cl.representative} (intra-cluster corr {cl.avg_intra_corr:.2f})",
                            cl.avg_intra_corr * 0.1, "medium",
                        ))

        # High crisis correlation shift
        for pc in self.pair_correlations:
            if pc.correlation_shift > 0.3:
                recs.append(Recommendation(
                    "reweight", pc.strategy_a,
                    f"Crisis correlation with {pc.strategy_b} spikes +{pc.correlation_shift:.2f}",
                    pc.correlation_shift * 0.05, "high",
                ))

        # Reweight to optimal
        if self.optimal_weights:
            for s in self.strategies:
                curr = self.weights.get(s, 0)
                opt = self.optimal_weights.weights.get(s, 0)
                diff = opt - curr
                if abs(diff) > 0.05:
                    action = "reweight"
                    recs.append(Recommendation(
                        action, s,
                        f"Current {curr:.0%} → optimal {opt:.0%} for max diversification",
                        abs(diff) * 0.5, "low",
                    ))

        return sorted(recs, key=lambda r: -r.estimated_impact)

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.diversification is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        return {
            "heatmap": self._chart_heatmap(),
            "dendrogram": self._chart_dendrogram(),
            "regime": self._chart_regime(),
            "risk": self._chart_risk(),
        }

    def _chart_heatmap(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        corr = self.returns.corr()
        n = len(corr)
        if n < 2:
            return ""
        fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
        im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.values[i,j]) > 0.5 else "black")
        fig.colorbar(im, shrink=0.8); ax.set_title("Strategy Correlation Heatmap", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_dendrogram(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro
        n = len(self.strategies)
        if n < 2:
            return ""
        corr = self.returns.corr().values.copy()
        dist = np.clip(1 - corr, 0, 2)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 4))
        scipy_dendro(Z, labels=self.strategies, ax=ax, leaf_rotation=45, leaf_font_size=8)
        ax.set_title("Strategy Dendrogram", fontsize=11)
        ax.set_ylabel("Distance (1 - corr)"); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_regime(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.regime_shifts:
            return ""
        names = [r.regime for r in self.regime_shifts]
        avgs = [r.avg_corr for r in self.regime_shifts]
        colors = {"bull": "#16a34a", "bear": "#dc2626", "high_vol": "#f59e0b", "neutral": "#64748b"}
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(names, avgs, color=[colors.get(n, "#3b82f6") for n in names], alpha=0.85)
        ax.set_ylabel("Avg Correlation"); ax.set_title("Correlation by Regime", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_risk(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.marginal_risks:
            return ""
        names = [m.strategy for m in self.marginal_risks]
        mcs = [m.marginal_contribution for m in self.marginal_risks]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
        colors = ["#dc2626" if mc > np.mean(mcs) else "#16a34a" for mc in mcs]
        ax.barh(names, mcs, color=colors, alpha=0.85)
        ax.set_xlabel("Marginal Risk Contribution"); ax.set_title("Risk Contribution", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        d = self.diversification or DiversificationMetrics(1, 1, 0, 0, 0)
        ow = self.optimal_weights or OptimalWeights({}, 1, 0, 0)

        pair_rows = ""
        for p in self.pair_correlations[:20]:
            shift_cls = "bad" if p.correlation_shift > 0.2 else ""
            pair_rows += (f'<tr><td>{p.strategy_a} / {p.strategy_b}</td><td>{p.full_corr:.3f}</td>'
                         f'<td>{p.current_corr:.3f}</td><td>{p.crisis_corr:.3f}</td>'
                         f'<td>{p.calm_corr:.3f}</td>'
                         f'<td class="{shift_cls}">{p.correlation_shift:+.3f}</td>'
                         f'<td>{p.tail_dependence:.3f}</td></tr>\n')

        risk_rows = ""
        for m in self.marginal_risks:
            risk_rows += (f'<tr><td>{m.strategy}</td><td>{m.weight:.1%}</td>'
                         f'<td>{m.marginal_contribution:.4f}</td>'
                         f'<td>{m.standalone_vol:.4f}</td><td>{m.beta_to_portfolio:.2f}</td></tr>\n')

        cluster_rows = ""
        for c in self.clusters:
            cluster_rows += (f'<tr><td>{c.cluster_id}</td><td>{", ".join(c.strategies)}</td>'
                            f'<td>{c.avg_intra_corr:.3f}</td><td>{c.representative}</td></tr>\n')

        regime_rows = ""
        for r in self.regime_shifts:
            cls = "bad" if r.shift_from_overall > 0.1 else "good" if r.shift_from_overall < -0.1 else ""
            regime_rows += (f'<tr><td>{r.regime}</td><td>{r.n_obs}</td><td>{r.avg_corr:.3f}</td>'
                           f'<td>{r.max_corr:.3f}</td><td class="{cls}">{r.shift_from_overall:+.3f}</td></tr>\n')

        rec_rows = ""
        for r in self.recommendations[:15]:
            cls = {"high": "bad", "medium": "warn", "low": ""}.get(r.priority, "")
            rec_rows += (f'<tr><td>{r.action.upper()}</td><td>{r.strategy}</td>'
                        f'<td style="text-align:left">{r.reason}</td>'
                        f'<td class="{cls}">{r.priority}</td></tr>\n')
        if not rec_rows:
            rec_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No recommendations</td></tr>'

        opt_rows = ""
        for s in sorted(ow.weights, key=lambda x: -ow.weights.get(x, 0)):
            w = ow.weights[s]
            opt_rows += f'<tr><td>{s}</td><td>{w:.1%}</td></tr>\n'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Strategy Correlation Analysis</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }} .warn {{ color:#f59e0b; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Strategy Correlation &amp; Diversification</h1>
<div class="meta">{len(self.strategies)} strategies &middot; {len(self.returns)} observations &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value">{d.diversification_ratio:.2f}</div><div class="label">Diversification Ratio</div></div>
  <div class="kpi"><div class="value">{d.effective_n:.1f}</div><div class="label">Effective N</div></div>
  <div class="kpi"><div class="value">{d.avg_correlation:.3f}</div><div class="label">Avg Correlation</div></div>
  <div class="kpi"><div class="value">{len(self.clusters)}</div><div class="label">Clusters</div></div>
  <div class="kpi"><div class="value">{len(self.recommendations)}</div><div class="label">Recommendations</div></div>
</div>
<h2>1. Correlation Heatmap</h2>{_img("heatmap")}
<table><thead><tr><th>Pair</th><th>Full</th><th>Current</th><th>Crisis</th><th>Calm</th><th>Shift</th><th>Tail Dep</th></tr></thead>
<tbody>{pair_rows}</tbody></table>
<h2>2. Strategy Clustering</h2>{_img("dendrogram")}
<table><thead><tr><th>Cluster</th><th>Strategies</th><th>Avg Intra Corr</th><th>Representative</th></tr></thead>
<tbody>{cluster_rows}</tbody></table>
<h2>3. Risk Contribution</h2>{_img("risk")}
<table><thead><tr><th>Strategy</th><th>Weight</th><th>Marginal Risk</th><th>Vol</th><th>Beta</th></tr></thead>
<tbody>{risk_rows}</tbody></table>
<h2>4. Regime Correlation Shifts</h2>{_img("regime")}
<table><thead><tr><th>Regime</th><th>Obs</th><th>Avg Corr</th><th>Max Corr</th><th>Shift</th></tr></thead>
<tbody>{regime_rows}</tbody></table>
<h2>5. Optimal Weights (Max Diversification)</h2>
<p>Diversification ratio: {ow.diversification_ratio:.2f} | Vol: {ow.portfolio_vol:.4f} | Sharpe: {ow.sharpe:.2f}</p>
<table><thead><tr><th>Strategy</th><th>Weight</th></tr></thead><tbody>{opt_rows}</tbody></table>
<h2>6. Recommendations</h2>
<table><thead><tr><th>Action</th><th>Strategy</th><th>Reason</th><th>Priority</th></tr></thead>
<tbody>{rec_rows}</tbody></table>
<footer>Generated by <code>compass/strategy_correlation.py</code></footer>
</body></html>"""
        return html
