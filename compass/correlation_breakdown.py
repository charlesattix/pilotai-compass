"""
Correlation breakdown analyzer for credit spread portfolios.

Detects contagion (correlation spikes during stress), computes conditional
correlation matrices by regime *and* volatility level, calculates a rolling
fragility timeline, tracks rolling conditional correlations, and quantifies
diversification benefits in normal vs stress periods.

Generates an HTML report at reports/correlation_breakdown.html with
conditional heatmaps, fragility timeline, and contagion risk indicators.

Usage::

    from compass.correlation_breakdown import CorrelationBreakdownAnalyzer
    analyzer = CorrelationBreakdownAnalyzer(returns_df, regimes=regime_series)
    results = analyzer.analyze()
    analyzer.generate_report("reports/correlation_breakdown.html")
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "correlation_breakdown.html"

REGIMES = ("bull", "bear", "high_vol", "neutral")
STRESS_REGIMES = ("bear", "high_vol")
CALM_REGIMES = ("bull", "neutral")
VOL_BUCKETS = ("low", "medium", "high")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class ContagionEvent:
    """A detected correlation spike indicative of contagion."""
    date: str
    pair: Tuple[str, str]
    baseline_corr: float
    stress_corr: float
    delta: float
    regime: str
    severity: str          # "low", "medium", "high"


@dataclass
class ContagionRiskIndicator:
    """Aggregate contagion risk assessment."""
    level: str             # "low", "medium", "high", "critical"
    n_events: int
    avg_delta: float
    max_delta: float
    pct_pairs_affected: float
    top_pair: Optional[Tuple[str, str]]


@dataclass
class RegimeCorrelation:
    """Conditional correlation matrix for a single regime."""
    regime: str
    matrix: np.ndarray
    mean_corr: float
    max_corr: float
    min_corr: float
    n_obs: int
    assets: List[str]


@dataclass
class VolCorrelation:
    """Conditional correlation matrix for a volatility bucket."""
    bucket: str            # "low", "medium", "high"
    matrix: np.ndarray
    mean_corr: float
    max_corr: float
    n_obs: int
    vol_range: Tuple[float, float]
    assets: List[str]


@dataclass
class FragilityScore:
    """Portfolio fragility assessment."""
    score: float               # 0-100, higher = more fragile
    eigenvalue_ratio: float    # first eigenvalue / sum
    mean_stress_corr: float
    mean_calm_corr: float
    correlation_gap: float     # stress - calm
    contagion_count: int
    diversification_ratio: float
    tail_concentration: float  # how much risk concentrates in tails


@dataclass
class FragilityTimepoint:
    """Fragility score at a single point in time."""
    date: str
    score: float
    eigenvalue_ratio: float
    mean_corr: float
    regime: str


@dataclass
class RollingCorrelation:
    """Rolling conditional correlation for a single asset pair."""
    pair: Tuple[str, str]
    dates: List[str]
    values: List[float]
    regime_labels: List[str]


@dataclass
class DiversificationBenefit:
    """Quantified diversification benefit comparing normal vs stress."""
    normal_portfolio_vol: float
    normal_weighted_avg_vol: float
    normal_benefit: float
    stress_portfolio_vol: float
    stress_weighted_avg_vol: float
    stress_benefit: float
    benefit_erosion: float     # how much diversification disappears in stress
    marginal_contributions: Dict[str, float]
    stress_marginal_contributions: Dict[str, float]


# ── Analyzer ────────────────────────────────────────────────────────────


class CorrelationBreakdownAnalyzer:
    """Full correlation breakdown analysis for a portfolio of return series."""

    def __init__(
        self,
        returns: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        window: int = 60,
        contagion_threshold: float = 0.3,
        weights: Optional[Dict[str, float]] = None,
        vol_column: Optional[str] = None,
    ) -> None:
        self.returns = returns.copy()
        self.assets = list(returns.columns)
        self.regimes = regimes if regimes is not None else pd.Series(
            "neutral", index=returns.index
        )
        self.window = window
        self.contagion_threshold = contagion_threshold
        n = len(self.assets)
        self.weights = weights or {a: 1.0 / n for a in self.assets}
        self.vol_column = vol_column

        # Compute realized vol for volatility bucketing
        self._realized_vol = self.returns.std(axis=1).rolling(
            min(20, max(3, len(returns) // 10))
        ).mean()

        # Results populated by analyze()
        self.contagion_events: List[ContagionEvent] = []
        self.contagion_risk: Optional[ContagionRiskIndicator] = None
        self.regime_correlations: Dict[str, RegimeCorrelation] = {}
        self.vol_correlations: Dict[str, VolCorrelation] = {}
        self.fragility: Optional[FragilityScore] = None
        self.fragility_timeline: List[FragilityTimepoint] = []
        self.rolling_correlations: List[RollingCorrelation] = []
        self.diversification: Optional[DiversificationBenefit] = None

    # ── Class constructors ──────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        returns_path: str,
        regimes_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "CorrelationBreakdownAnalyzer":
        """Load from CSV files."""
        ret = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        reg = None
        if regimes_path:
            reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
            reg = reg_df.iloc[:, 0]
        return cls(ret, regimes=reg, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis and return results dict."""
        self.regime_correlations = self._conditional_correlations_by_regime()
        self.vol_correlations = self._conditional_correlations_by_vol()
        self.contagion_events = self._detect_contagion()
        self.contagion_risk = self._contagion_risk_indicator()
        self.rolling_correlations = self._rolling_conditional_correlations()
        self.diversification = self._diversification_benefit()
        self.fragility = self._fragility_score()
        self.fragility_timeline = self._rolling_fragility()
        return {
            "contagion_events": self.contagion_events,
            "contagion_risk": self.contagion_risk,
            "regime_correlations": self.regime_correlations,
            "vol_correlations": self.vol_correlations,
            "fragility": self.fragility,
            "fragility_timeline": self.fragility_timeline,
            "rolling_correlations": self.rolling_correlations,
            "diversification": self.diversification,
        }

    # ── Conditional correlation matrices by regime ──────────────────────

    def _conditional_correlations_by_regime(self) -> Dict[str, RegimeCorrelation]:
        """Compute correlation matrix for each regime."""
        results: Dict[str, RegimeCorrelation] = {}
        for regime in REGIMES:
            mask = self.regimes == regime
            subset = self.returns.loc[mask]
            if len(subset) < 3:
                continue
            matrix = subset.corr().values.copy()
            off_diag = matrix.copy()
            np.fill_diagonal(off_diag, np.nan)
            results[regime] = RegimeCorrelation(
                regime=regime,
                matrix=matrix,
                mean_corr=float(np.nanmean(off_diag)),
                max_corr=float(np.nanmax(off_diag)),
                min_corr=float(np.nanmin(off_diag)),
                n_obs=len(subset),
                assets=self.assets,
            )
        return results

    # ── Conditional correlation matrices by volatility level ────────────

    def _conditional_correlations_by_vol(self) -> Dict[str, VolCorrelation]:
        """Compute correlation matrix for low/medium/high volatility buckets."""
        vol = self._realized_vol.dropna()
        if len(vol) < 10:
            return {}
        terciles = vol.quantile([1 / 3, 2 / 3]).values
        buckets = {
            "low": vol <= terciles[0],
            "medium": (vol > terciles[0]) & (vol <= terciles[1]),
            "high": vol > terciles[1],
        }
        ranges = {
            "low": (float(vol.min()), float(terciles[0])),
            "medium": (float(terciles[0]), float(terciles[1])),
            "high": (float(terciles[1]), float(vol.max())),
        }
        results: Dict[str, VolCorrelation] = {}
        for bucket, mask in buckets.items():
            idx = vol.index[mask]
            subset = self.returns.loc[self.returns.index.isin(idx)]
            if len(subset) < 3:
                continue
            matrix = subset.corr().values.copy()
            off_diag = matrix.copy()
            np.fill_diagonal(off_diag, np.nan)
            results[bucket] = VolCorrelation(
                bucket=bucket,
                matrix=matrix,
                mean_corr=float(np.nanmean(off_diag)),
                max_corr=float(np.nanmax(off_diag)),
                n_obs=len(subset),
                vol_range=ranges[bucket],
                assets=self.assets,
            )
        return results

    # ── Contagion detection ─────────────────────────────────────────────

    def _detect_contagion(self) -> List[ContagionEvent]:
        """Detect correlation spikes: stress-regime corr >> calm-regime corr."""
        calm = self._best_regime_corr(CALM_REGIMES)
        stress = self._best_regime_corr(STRESS_REGIMES)
        if calm is None or stress is None:
            return []

        events: List[ContagionEvent] = []
        n = len(self.assets)
        for i in range(n):
            for j in range(i + 1, n):
                base_c = calm.matrix[i, j]
                stress_c = stress.matrix[i, j]
                delta = stress_c - base_c
                if delta >= self.contagion_threshold:
                    mask = self.regimes == stress.regime
                    stress_dates = self.returns.index[mask]
                    peak_date = str(stress_dates[-1]) if len(stress_dates) > 0 else ""
                    severity = (
                        "high" if delta >= 0.6
                        else "medium" if delta >= 0.4
                        else "low"
                    )
                    events.append(ContagionEvent(
                        date=peak_date,
                        pair=(self.assets[i], self.assets[j]),
                        baseline_corr=float(base_c),
                        stress_corr=float(stress_c),
                        delta=float(delta),
                        regime=stress.regime,
                        severity=severity,
                    ))
        return sorted(events, key=lambda e: -e.delta)

    def _contagion_risk_indicator(self) -> ContagionRiskIndicator:
        """Aggregate contagion risk into a single indicator."""
        n_pairs = len(self.assets) * (len(self.assets) - 1) // 2
        n_events = len(self.contagion_events)
        if n_events == 0:
            return ContagionRiskIndicator(
                level="low", n_events=0, avg_delta=0.0, max_delta=0.0,
                pct_pairs_affected=0.0, top_pair=None,
            )
        deltas = [e.delta for e in self.contagion_events]
        avg_d = float(np.mean(deltas))
        max_d = float(np.max(deltas))
        pct = n_events / max(n_pairs, 1)
        top_pair = self.contagion_events[0].pair

        if pct >= 0.5 and avg_d >= 0.5:
            level = "critical"
        elif pct >= 0.3 or avg_d >= 0.4:
            level = "high"
        elif pct >= 0.15 or avg_d >= 0.3:
            level = "medium"
        else:
            level = "low"

        return ContagionRiskIndicator(
            level=level, n_events=n_events, avg_delta=avg_d,
            max_delta=max_d, pct_pairs_affected=pct, top_pair=top_pair,
        )

    # ── Rolling conditional correlation ─────────────────────────────────

    def _rolling_conditional_correlations(self) -> List[RollingCorrelation]:
        """Compute rolling correlations for all asset pairs."""
        results: List[RollingCorrelation] = []
        n = len(self.assets)
        if len(self.returns) < self.window:
            return results

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.assets[i], self.assets[j]
                rolling = self.returns[[a, b]].rolling(self.window).corr()
                pair_corr = rolling.unstack().iloc[:, 1].dropna()
                dates = [str(d) for d in pair_corr.index]
                vals = pair_corr.values.tolist()
                regime_labels = [
                    str(self.regimes.get(d, "neutral"))
                    for d in pair_corr.index
                ]
                results.append(RollingCorrelation(
                    pair=(a, b), dates=dates, values=vals,
                    regime_labels=regime_labels,
                ))
        return results

    # ── Diversification benefit: normal vs stress ───────────────────────

    def _diversification_benefit(self) -> DiversificationBenefit:
        """Quantify diversification benefit in normal vs stress periods."""
        w = np.array([self.weights.get(a, 0) for a in self.assets])
        w = w / w.sum()

        stress_mask = self.regimes.isin(STRESS_REGIMES)
        calm_mask = self.regimes.isin(CALM_REGIMES)

        normal_ret = self.returns.loc[calm_mask]
        stress_ret = self.returns.loc[stress_mask]

        def _compute(ret_slice: pd.DataFrame) -> Tuple[float, float, float, Dict[str, float]]:
            if len(ret_slice) < 3:
                return 0.0, 0.0, 0.0, {a: 0.0 for a in self.assets}
            cov = ret_slice.cov().values
            pvar = float(w @ cov @ w)
            pvol = float(np.sqrt(max(pvar, 0)))
            ivols = np.sqrt(np.diag(cov))
            wavg = float(np.dot(w, ivols))
            benefit = 1.0 - (pvol / wavg) if wavg > 0 else 0.0
            mc: Dict[str, float] = {}
            if pvol > 0:
                mcv = (cov @ w) / pvol
                for k, a in enumerate(self.assets):
                    mc[a] = float(w[k] * mcv[k])
            else:
                mc = {a: 0.0 for a in self.assets}
            return pvol, wavg, benefit, mc

        n_pvol, n_wavg, n_ben, n_mc = _compute(normal_ret)
        s_pvol, s_wavg, s_ben, s_mc = _compute(stress_ret)
        erosion = n_ben - s_ben if n_ben > 0 else 0.0

        return DiversificationBenefit(
            normal_portfolio_vol=n_pvol,
            normal_weighted_avg_vol=n_wavg,
            normal_benefit=n_ben,
            stress_portfolio_vol=s_pvol,
            stress_weighted_avg_vol=s_wavg,
            stress_benefit=s_ben,
            benefit_erosion=erosion,
            marginal_contributions=n_mc,
            stress_marginal_contributions=s_mc,
        )

    # ── Fragility score ─────────────────────────────────────────────────

    def _fragility_score(self) -> FragilityScore:
        """Compute portfolio fragility score (0-100)."""
        corr_matrix = self.returns.corr().values.copy()
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eig_ratio = float(eigenvalues[0] / eigenvalues.sum()) if eigenvalues.sum() > 0 else 0.0

        stress_rc = self._best_regime_corr(STRESS_REGIMES)
        calm_rc = self._best_regime_corr(CALM_REGIMES)
        mean_stress = stress_rc.mean_corr if stress_rc else 0.0
        mean_calm = calm_rc.mean_corr if calm_rc else 0.0
        corr_gap = max(mean_stress - mean_calm, 0.0)

        contagion_count = len(self.contagion_events)
        div = self.diversification
        div_ratio = div.normal_benefit if div else 0.0
        erosion = div.benefit_erosion if div else 0.0

        # Tail concentration: eigenvalue ratio in stress periods
        tail_conc = 0.0
        if stress_rc is not None and stress_rc.matrix.shape[0] > 1:
            s_eig = np.linalg.eigvalsh(stress_rc.matrix)
            s_eig = np.sort(s_eig)[::-1]
            tail_conc = float(s_eig[0] / s_eig.sum()) if s_eig.sum() > 0 else 0.0

        raw = (
            20.0 * eig_ratio
            + 20.0 * max(mean_stress, 0)
            + 15.0 * corr_gap
            + 15.0 * (1.0 - div_ratio)
            + 15.0 * erosion
            + 10.0 * min(contagion_count / max(len(self.assets), 1), 1.0)
            + 5.0 * tail_conc
        )
        score = max(0.0, min(100.0, raw))

        return FragilityScore(
            score=score,
            eigenvalue_ratio=eig_ratio,
            mean_stress_corr=mean_stress,
            mean_calm_corr=mean_calm,
            correlation_gap=corr_gap,
            contagion_count=contagion_count,
            diversification_ratio=div_ratio,
            tail_concentration=tail_conc,
        )

    # ── Rolling fragility timeline ──────────────────────────────────────

    def _rolling_fragility(self) -> List[FragilityTimepoint]:
        """Compute fragility score over rolling windows for a timeline."""
        timeline: List[FragilityTimepoint] = []
        step = max(1, self.window // 4)
        if len(self.returns) < self.window:
            return timeline

        for end in range(self.window, len(self.returns), step):
            start = end - self.window
            window_ret = self.returns.iloc[start:end]
            corr_matrix = window_ret.corr().values.copy()
            off_diag = corr_matrix.copy()
            np.fill_diagonal(off_diag, np.nan)
            mean_c = float(np.nanmean(off_diag))

            eigvals = np.linalg.eigvalsh(corr_matrix)
            eigvals = np.sort(eigvals)[::-1]
            eig_r = float(eigvals[0] / eigvals.sum()) if eigvals.sum() > 0 else 0.0

            raw_score = 50.0 * eig_r + 50.0 * max(mean_c, 0)
            score = max(0.0, min(100.0, raw_score))

            date_idx = self.returns.index[end - 1]
            regime = str(self.regimes.get(date_idx, "neutral"))

            timeline.append(FragilityTimepoint(
                date=str(date_idx),
                score=score,
                eigenvalue_ratio=eig_r,
                mean_corr=mean_c,
                regime=regime,
            ))
        return timeline

    # ── Helpers ──────────────────────────────────────────────────────────

    def _best_regime_corr(
        self, candidates: Tuple[str, ...],
    ) -> Optional[RegimeCorrelation]:
        """Return first available RegimeCorrelation from candidates."""
        for r in candidates:
            if r in self.regime_correlations:
                return self.regime_correlations[r]
        return None

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report. Runs analyze() if not yet run."""
        if self.fragility is None:
            self.analyze()

        charts = self._render_charts()
        html = self._build_html(charts)
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

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["regime_heatmap"] = self._chart_regime_heatmaps()
        charts["vol_heatmap"] = self._chart_vol_heatmaps()
        charts["rolling"] = self._chart_rolling_correlation()
        charts["fragility_timeline"] = self._chart_fragility_timeline()
        charts["diversification"] = self._chart_diversification()
        charts["contagion"] = self._chart_contagion_network()
        return charts

    def _chart_regime_heatmaps(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        regimes = [r for r in REGIMES if r in self.regime_correlations]
        if not regimes:
            return ""
        ncols = len(regimes)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]
        for ax, regime in zip(axes, regimes):
            rc = self.regime_correlations[regime]
            im = ax.imshow(rc.matrix, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(rc.assets)))
            ax.set_xticklabels(rc.assets, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(rc.assets)))
            ax.set_yticklabels(rc.assets, fontsize=7)
            ax.set_title(f"{regime} (n={rc.n_obs})\nμ={rc.mean_corr:.2f}", fontsize=9)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle("Conditional Correlation by Regime", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_vol_heatmaps(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        buckets = [b for b in VOL_BUCKETS if b in self.vol_correlations]
        if not buckets:
            return ""
        ncols = len(buckets)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]
        for ax, bucket in zip(axes, buckets):
            vc = self.vol_correlations[bucket]
            im = ax.imshow(vc.matrix, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(vc.assets)))
            ax.set_xticklabels(vc.assets, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(vc.assets)))
            ax.set_yticklabels(vc.assets, fontsize=7)
            ax.set_title(f"{bucket} vol (n={vc.n_obs})\nμ={vc.mean_corr:.2f}", fontsize=9)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle("Conditional Correlation by Volatility Level", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_rolling_correlation(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.rolling_correlations:
            return ""
        n_pairs = min(len(self.rolling_correlations), 6)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
        if n_pairs == 1:
            axes = [axes]
        regime_colors = {
            "bull": "#16a34a", "bear": "#dc2626",
            "high_vol": "#f59e0b", "neutral": "#64748b",
        }
        for ax, rc in zip(axes, self.rolling_correlations[:n_pairs]):
            vals = np.array(rc.values)
            xs = range(len(vals))
            ax.plot(xs, vals, color="#334155", lw=0.8, alpha=0.9)
            for k in range(len(rc.regime_labels) - 1):
                color = regime_colors.get(rc.regime_labels[k], "#f8fafc")
                ax.axvspan(k, k + 1, alpha=0.08, color=color)
            ax.set_ylabel(f"{rc.pair[0]}/{rc.pair[1]}", fontsize=8)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", lw=0.3)
            ax.grid(True, alpha=0.2)
        fig.suptitle("Rolling Conditional Correlations", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_fragility_timeline(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.fragility_timeline:
            return ""
        dates = list(range(len(self.fragility_timeline)))
        scores = [fp.score for fp in self.fragility_timeline]
        regimes = [fp.regime for fp in self.fragility_timeline]
        regime_colors = {
            "bull": "#16a34a", "bear": "#dc2626",
            "high_vol": "#f59e0b", "neutral": "#64748b",
        }

        fig, ax = plt.subplots(figsize=(10, 3.5))
        # Background shading by regime
        for k in range(len(dates) - 1):
            color = regime_colors.get(regimes[k], "#f8fafc")
            ax.axvspan(dates[k], dates[k + 1], alpha=0.1, color=color)
        # Fragility line with gradient coloring
        for k in range(len(dates) - 1):
            c = "#16a34a" if scores[k] < 33 else "#f59e0b" if scores[k] < 66 else "#dc2626"
            ax.plot(dates[k:k + 2], scores[k:k + 2], color=c, lw=1.5)
        # Threshold lines
        ax.axhline(33, color="#16a34a", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(66, color="#dc2626", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Fragility Score", fontsize=9)
        ax.set_title("Fragility Timeline", fontsize=11)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_contagion_network(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.contagion_events:
            return ""
        fig, ax = plt.subplots(figsize=(6, 4))
        pairs = [(e.pair, e.delta, e.severity) for e in self.contagion_events[:10]]
        labels = [f"{p[0]}/{p[1]}" for p, _, _ in pairs]
        deltas = [d for _, d, _ in pairs]
        colors = {"high": "#dc2626", "medium": "#f59e0b", "low": "#64748b"}
        bar_colors = [colors.get(s, "#64748b") for _, _, s in pairs]
        y_pos = range(len(labels))
        ax.barh(y_pos, deltas, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Correlation Delta (Stress − Calm)")
        ax.set_title("Contagion: Largest Correlation Spikes", fontsize=11)
        ax.axvline(self.contagion_threshold, color="#dc2626", lw=0.8, ls="--", alpha=0.6)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_diversification(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.diversification is None:
            return ""
        d = self.diversification
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Left: normal vs stress diversification
        categories = ["Normal", "Stress"]
        benefits = [d.normal_benefit, d.stress_benefit]
        colors = ["#16a34a", "#dc2626"]
        ax1.bar(categories, benefits, color=colors, alpha=0.85, edgecolor="white")
        ax1.set_ylabel("Diversification Benefit")
        ax1.set_title(f"Benefit Erosion: {d.benefit_erosion:.1%}", fontsize=10)
        ax1.set_ylim(min(0, min(benefits) - 0.05), max(benefits) + 0.1)
        ax1.grid(True, axis="y", alpha=0.3)

        # Right: marginal contributions comparison
        assets = list(d.marginal_contributions.keys())
        normal_mc = [d.marginal_contributions.get(a, 0) for a in assets]
        stress_mc = [d.stress_marginal_contributions.get(a, 0) for a in assets]
        x = np.arange(len(assets))
        width = 0.35
        ax2.bar(x - width / 2, normal_mc, width, label="Normal", color="#16a34a", alpha=0.7)
        ax2.bar(x + width / 2, stress_mc, width, label="Stress", color="#dc2626", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(assets, fontsize=7, rotation=45, ha="right")
        ax2.set_ylabel("Marginal Risk Contribution")
        ax2.set_title("Risk Contribution Shift", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        frag = self.fragility or FragilityScore(0, 0, 0, 0, 0, 0, 0, 0)
        div = self.diversification or DiversificationBenefit(0, 0, 0, 0, 0, 0, 0, {}, {})
        risk = self.contagion_risk or ContagionRiskIndicator("low", 0, 0, 0, 0, None)

        # Contagion risk indicator color
        risk_colors = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626", "critical": "#7f1d1d"}
        risk_color = risk_colors.get(risk.level, "#64748b")

        # Contagion table
        contagion_rows = ""
        for e in self.contagion_events:
            sev_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#64748b"}[e.severity]
            contagion_rows += (
                f'<tr><td>{e.pair[0]} / {e.pair[1]}</td>'
                f'<td>{e.baseline_corr:.3f}</td>'
                f'<td class="bad">{e.stress_corr:.3f}</td>'
                f'<td class="bad">+{e.delta:.3f}</td>'
                f'<td style="color:{sev_color};font-weight:600">{e.severity.upper()}</td>'
                f'<td>{e.regime}</td></tr>\n'
            )
        if not contagion_rows:
            contagion_rows = '<tr><td colspan="6" style="text-align:center;color:#64748b">No contagion events detected</td></tr>'

        # Regime correlation summary table
        regime_rows = ""
        for regime in REGIMES:
            rc = self.regime_correlations.get(regime)
            if rc is None:
                continue
            cls = "bad" if rc.mean_corr > 0.5 else "good" if rc.mean_corr < 0.3 else ""
            regime_rows += (
                f'<tr><td>{rc.regime}</td><td>{rc.n_obs}</td>'
                f'<td class="{cls}">{rc.mean_corr:.3f}</td>'
                f'<td>{rc.max_corr:.3f}</td>'
                f'<td>{rc.min_corr:.3f}</td></tr>\n'
            )

        # Vol correlation table
        vol_rows = ""
        for bucket in VOL_BUCKETS:
            vc = self.vol_correlations.get(bucket)
            if vc is None:
                continue
            cls = "bad" if vc.mean_corr > 0.5 else "good" if vc.mean_corr < 0.3 else ""
            vol_rows += (
                f'<tr><td>{vc.bucket}</td><td>{vc.n_obs}</td>'
                f'<td class="{cls}">{vc.mean_corr:.3f}</td>'
                f'<td>{vc.max_corr:.3f}</td>'
                f'<td>{vc.vol_range[0]:.4f}–{vc.vol_range[1]:.4f}</td></tr>\n'
            )

        # Diversification table
        div_rows = ""
        if div.marginal_contributions:
            for a in sorted(div.marginal_contributions, key=lambda x: -div.marginal_contributions[x]):
                n_mc = div.marginal_contributions[a]
                s_mc = div.stress_marginal_contributions.get(a, 0)
                shift = s_mc - n_mc
                shift_cls = "bad" if shift > 0 else "good"
                div_rows += (
                    f'<tr><td>{a}</td><td>{n_mc:.4f}</td>'
                    f'<td>{s_mc:.4f}</td>'
                    f'<td class="{shift_cls}">{shift:+.4f}</td></tr>\n'
                )

        frag_cls = "bad" if frag.score >= 66 else "good" if frag.score < 33 else ""

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Correlation Breakdown Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  h3 {{ color: #475569; margin-top: 1.2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .warn {{ color: #f59e0b; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .risk-badge {{ display: inline-block; padding: 0.3em 0.8em; border-radius: 4px;
                 color: white; font-weight: 700; font-size: 0.9em; }}
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

<h1>Correlation Breakdown Analysis</h1>
<div class="meta">{len(self.assets)} assets &middot; {len(self.returns)} observations &middot; Window {self.window}d &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value {frag_cls}">{frag.score:.1f}</div><div class="label">Fragility Score</div></div>
  <div class="kpi"><div class="value"><span class="risk-badge" style="background:{risk_color}">{risk.level.upper()}</span></div><div class="label">Contagion Risk</div></div>
  <div class="kpi"><div class="value">{div.normal_benefit:.1%}</div><div class="label">Normal Diversification</div></div>
  <div class="kpi"><div class="value {'' if div.stress_benefit > 0.1 else 'bad'}">{div.stress_benefit:.1%}</div><div class="label">Stress Diversification</div></div>
  <div class="kpi"><div class="value bad">{div.benefit_erosion:.1%}</div><div class="label">Benefit Erosion</div></div>
  <div class="kpi"><div class="value">{len(self.contagion_events)}</div><div class="label">Contagion Events</div></div>
</div>

<h2>1. Contagion Detection</h2>
<p>Asset pairs where stress-regime correlation exceeds calm-regime by &ge; {self.contagion_threshold:.0%}.
   <strong>Risk Level: <span style="color:{risk_color}">{risk.level.upper()}</span></strong>
   &mdash; {risk.pct_pairs_affected:.0%} of pairs affected, avg &Delta; = {risk.avg_delta:.3f}</p>
{_img("contagion")}
<table>
<thead><tr><th>Pair</th><th>Baseline</th><th>Stress</th><th>Delta</th><th>Severity</th><th>Regime</th></tr></thead>
<tbody>{contagion_rows}</tbody>
</table>

<h2>2. Conditional Correlation by Regime</h2>
{_img("regime_heatmap")}
<table>
<thead><tr><th>Regime</th><th>Obs</th><th>Mean Corr</th><th>Max Corr</th><th>Min Corr</th></tr></thead>
<tbody>{regime_rows}</tbody>
</table>

<h2>3. Conditional Correlation by Volatility Level</h2>
{_img("vol_heatmap")}
<table>
<thead><tr><th>Vol Bucket</th><th>Obs</th><th>Mean Corr</th><th>Max Corr</th><th>Vol Range</th></tr></thead>
<tbody>{vol_rows}</tbody>
</table>

<h2>4. Fragility Timeline</h2>
{_img("fragility_timeline")}
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Fragility Score</td><td class="{frag_cls}">{frag.score:.1f} / 100</td></tr>
<tr><td>Eigenvalue Concentration</td><td>{frag.eigenvalue_ratio:.3f}</td></tr>
<tr><td>Mean Stress Correlation</td><td>{frag.mean_stress_corr:.3f}</td></tr>
<tr><td>Mean Calm Correlation</td><td>{frag.mean_calm_corr:.3f}</td></tr>
<tr><td>Correlation Gap (Stress − Calm)</td><td class="{"bad" if frag.correlation_gap > 0.2 else ""}">{frag.correlation_gap:.3f}</td></tr>
<tr><td>Tail Concentration</td><td>{frag.tail_concentration:.3f}</td></tr>
<tr><td>Contagion Events</td><td>{frag.contagion_count}</td></tr>
</tbody>
</table>

<h2>5. Rolling Conditional Correlations</h2>
{_img("rolling")}

<h2>6. Diversification Benefit (Normal vs Stress)</h2>
{_img("diversification")}
<table>
<thead><tr><th>Period</th><th>Portfolio Vol</th><th>Weighted Avg Vol</th><th>Benefit</th></tr></thead>
<tbody>
<tr><td>Normal</td><td>{div.normal_portfolio_vol:.4f}</td><td>{div.normal_weighted_avg_vol:.4f}</td><td class="good">{div.normal_benefit:.1%}</td></tr>
<tr><td>Stress</td><td>{div.stress_portfolio_vol:.4f}</td><td>{div.stress_weighted_avg_vol:.4f}</td><td class="{"good" if div.stress_benefit > 0.1 else "bad"}">{div.stress_benefit:.1%}</td></tr>
<tr><td><strong>Erosion</strong></td><td colspan="2"></td><td class="bad">{div.benefit_erosion:.1%}</td></tr>
</tbody>
</table>
<h3>Marginal Risk Contributions</h3>
<table>
<thead><tr><th>Asset</th><th>Normal</th><th>Stress</th><th>Shift</th></tr></thead>
<tbody>{div_rows}</tbody>
</table>

<footer>Generated by <code>compass/correlation_breakdown.py</code></footer>
</body></html>"""
        return html
