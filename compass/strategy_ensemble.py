"""
Strategy ensemble combiner for credit spread experiments.

Combines signals from multiple experiments (EXP-400, EXP-401, EXP-503,
EXP-600) using voting, stacking, or Bayesian model averaging.  Supports
dynamic weight adjustment based on recent performance, disagreement detection,
ensemble confidence scoring, and regime-conditional combining.

Generates an HTML report at reports/strategy_ensemble.html with weight
evolution, agreement heatmap, and performance attribution.

Usage::

    from compass.strategy_ensemble import StrategyEnsemble
    ensemble = StrategyEnsemble(signals_df, returns_df, regimes=regime_series)
    results = ensemble.analyze()
    ensemble.generate_report("reports/strategy_ensemble.html")
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
DEFAULT_OUTPUT = ROOT / "reports" / "strategy_ensemble.html"

DEFAULT_STRATEGIES = ("EXP-400", "EXP-401", "EXP-503", "EXP-600")
REGIMES = ("bull", "bear", "high_vol", "neutral")
COMBINE_METHODS = ("voting", "stacking", "bayesian")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class StrategyWeight:
    """Weight for a single strategy at a point in time."""
    strategy: str
    weight: float
    recent_sharpe: float
    recent_win_rate: float
    n_trades: int


@dataclass
class RegimeWeights:
    """Optimal strategy weights for a single regime."""
    regime: str
    weights: Dict[str, float]
    n_obs: int
    ensemble_sharpe: float


@dataclass
class DisagreementEvent:
    """Period where strategies produce conflicting signals."""
    date: str
    signals: Dict[str, float]
    agreement_score: float      # 0 = total disagreement, 1 = unanimous
    recommended_sizing: float   # fraction of normal size
    regime: str


@dataclass
class EnsembleConfidence:
    """Confidence metrics for the ensemble signal."""
    score: float               # 0-1
    agreement_ratio: float     # fraction of strategies agreeing with ensemble
    signal_dispersion: float   # std of strategy signals
    weight_concentration: float  # HHI of weights (1/n = even, 1 = concentrated)
    regime_stability: float    # consistency of weights across regimes


@dataclass
class PerformanceAttribution:
    """Performance attribution for a single strategy."""
    strategy: str
    total_return: float
    contribution: float        # weighted contribution to ensemble
    hit_rate: float
    avg_signal_strength: float
    correlation_with_ensemble: float


@dataclass
class WeightSnapshot:
    """Weights at a single point in time for the evolution chart."""
    date: str
    weights: Dict[str, float]
    ensemble_signal: float
    regime: str


# ── Ensemble combiner ──────────────────────────────────────────────────


class StrategyEnsemble:
    """Combine signals from multiple strategy experiments."""

    def __init__(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        regimes: Optional[pd.Series] = None,
        method: str = "bayesian",
        lookback: int = 60,
        halflife: int = 20,
        disagreement_threshold: float = 0.5,
        min_confidence: float = 0.3,
    ) -> None:
        self.signals = signals.copy()
        self.strategies = list(signals.columns)
        self.returns = returns.copy()
        self.regimes = regimes if regimes is not None else pd.Series(
            "neutral", index=signals.index,
        )
        if method not in COMBINE_METHODS:
            raise ValueError(f"method must be one of {COMBINE_METHODS}, got {method!r}")
        self.method = method
        self.lookback = lookback
        self.halflife = halflife
        self.disagreement_threshold = disagreement_threshold
        self.min_confidence = min_confidence

        # Align indexes
        common = signals.index.intersection(returns.index)
        self.signals = self.signals.loc[common]
        self.returns = self.returns.loc[common]
        self.regimes = self.regimes.reindex(common, fill_value="neutral")

        n = len(self.strategies)
        self._equal_weights = {s: 1.0 / n for s in self.strategies}

        # Results
        self.current_weights: Dict[str, StrategyWeight] = {}
        self.regime_weights: Dict[str, RegimeWeights] = {}
        self.disagreements: List[DisagreementEvent] = []
        self.confidence: Optional[EnsembleConfidence] = None
        self.attributions: List[PerformanceAttribution] = []
        self.weight_history: List[WeightSnapshot] = []
        self.ensemble_signals: Optional[pd.Series] = None

    # ── Class constructors ──────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        signals_path: str,
        returns_path: str,
        regimes_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "StrategyEnsemble":
        """Load from CSV files."""
        sig = pd.read_csv(signals_path, index_col=0, parse_dates=True)
        ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).squeeze("columns")
        reg = None
        if regimes_path:
            reg_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
            reg = reg_df.iloc[:, 0]
        return cls(sig, ret, regimes=reg, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run full ensemble analysis."""
        self.current_weights = self._compute_dynamic_weights()
        self.regime_weights = self._regime_conditional_weights()
        self.ensemble_signals = self._combine_signals()
        self.disagreements = self._detect_disagreements()
        self.confidence = self._ensemble_confidence()
        self.attributions = self._performance_attribution()
        self.weight_history = self._rolling_weight_evolution()
        return {
            "current_weights": self.current_weights,
            "regime_weights": self.regime_weights,
            "ensemble_signals": self.ensemble_signals,
            "disagreements": self.disagreements,
            "confidence": self.confidence,
            "attributions": self.attributions,
            "weight_history": self.weight_history,
        }

    def predict(self, signals: pd.Series) -> Tuple[float, float]:
        """Predict ensemble signal + confidence for a single observation.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals keyed by strategy name.

        Returns
        -------
        (ensemble_signal, sizing_factor) where sizing_factor is reduced
        when strategies disagree.
        """
        if self.confidence is None:
            self.analyze()

        # Get regime-conditional or dynamic weights
        regime = str(self.regimes.iloc[-1]) if len(self.regimes) > 0 else "neutral"
        rw = self.regime_weights.get(regime)
        weights = rw.weights if rw is not None else {
            s: self.current_weights[s].weight for s in self.strategies
        }

        ens_signal = sum(weights.get(s, 0) * signals.get(s, 0) for s in self.strategies)

        # Sizing: reduce when strategies disagree
        nonzero = signals[signals != 0]
        if len(nonzero) >= 2:
            majority = np.sign(nonzero.sum())
            agreement = float((np.sign(nonzero) == majority).mean()) if majority != 0 else 0.0
        else:
            agreement = 1.0
        sizing = max(self.min_confidence, agreement)

        return float(ens_signal), float(sizing)

    # ── Dynamic weight computation ──────────────────────────────────────

    def _compute_dynamic_weights(self) -> Dict[str, StrategyWeight]:
        """Compute exponentially weighted performance-based weights."""
        exp_weights = self._exponential_weights(len(self.returns))
        results: Dict[str, StrategyWeight] = {}
        raw_scores: Dict[str, float] = {}

        for strat in self.strategies:
            sig = self.signals[strat]
            # Strategy return: signal * actual return (correct direction = positive)
            strat_ret = sig * self.returns
            weighted_ret = strat_ret * exp_weights

            recent = strat_ret.iloc[-self.lookback:] if len(strat_ret) >= self.lookback else strat_ret
            sharpe = self._sharpe(recent)
            win_rate = float((recent > 0).mean()) if len(recent) > 0 else 0.0
            n_trades = int((sig.abs() > 0).sum())

            # Score combines Sharpe and consistency
            score = max(sharpe, 0.0) + 0.5 * win_rate
            raw_scores[strat] = score

            results[strat] = StrategyWeight(
                strategy=strat, weight=0.0,
                recent_sharpe=sharpe, recent_win_rate=win_rate,
                n_trades=n_trades,
            )

        # Normalize to sum to 1
        total = sum(raw_scores.values())
        if total > 0:
            for s in self.strategies:
                results[s].weight = raw_scores[s] / total
        else:
            for s in self.strategies:
                results[s].weight = 1.0 / len(self.strategies)

        return results

    def _exponential_weights(self, n: int) -> pd.Series:
        """Generate exponential decay weights for n observations."""
        decay = np.log(2) / self.halflife
        w = np.exp(-decay * np.arange(n)[::-1])
        w /= w.sum()
        return pd.Series(w, index=self.returns.index[:n])

    # ── Regime-conditional weights ──────────────────────────────────────

    def _regime_conditional_weights(self) -> Dict[str, RegimeWeights]:
        """Compute optimal weights per regime."""
        results: Dict[str, RegimeWeights] = {}
        for regime in REGIMES:
            mask = self.regimes == regime
            if mask.sum() < 5:
                continue
            regime_ret = self.returns.loc[mask]
            regime_sig = self.signals.loc[mask]

            raw: Dict[str, float] = {}
            for strat in self.strategies:
                strat_ret = regime_sig[strat] * regime_ret
                raw[strat] = max(self._sharpe(strat_ret), 0.0) + 0.01
            total = sum(raw.values())
            weights = {s: raw[s] / total for s in self.strategies}

            # Ensemble Sharpe for this regime
            w_arr = np.array([weights[s] for s in self.strategies])
            combined = sum(
                weights[s] * regime_sig[s] * regime_ret
                for s in self.strategies
            )
            ens_sharpe = self._sharpe(combined)

            results[regime] = RegimeWeights(
                regime=regime, weights=weights,
                n_obs=int(mask.sum()), ensemble_sharpe=ens_sharpe,
            )
        return results

    # ── Signal combining ────────────────────────────────────────────────

    def _combine_signals(self) -> pd.Series:
        """Combine strategy signals using the configured method."""
        if self.method == "voting":
            return self._combine_voting()
        elif self.method == "stacking":
            return self._combine_stacking()
        else:
            return self._combine_bayesian()

    def _combine_voting(self) -> pd.Series:
        """Majority vote: sign of mean signal."""
        return np.sign(self.signals.mean(axis=1))

    def _combine_stacking(self) -> pd.Series:
        """Weighted average using dynamic weights."""
        w = {s: self.current_weights[s].weight for s in self.strategies}
        combined = sum(w[s] * self.signals[s] for s in self.strategies)
        return combined

    def _combine_bayesian(self) -> pd.Series:
        """Bayesian model averaging: regime-conditional weighted average."""
        result = pd.Series(0.0, index=self.signals.index)
        for regime in REGIMES:
            mask = self.regimes == regime
            if mask.sum() == 0:
                continue
            rw = self.regime_weights.get(regime)
            if rw is None:
                weights = self._equal_weights
            else:
                weights = rw.weights
            for s in self.strategies:
                result.loc[mask] += weights[s] * self.signals.loc[mask, s]

        # Fill any remaining neutral rows (regimes not in REGIMES)
        unfilled = result.index.difference(
            self.regimes[self.regimes.isin(REGIMES)].index
        )
        if len(unfilled) > 0:
            for s in self.strategies:
                w = self.current_weights[s].weight
                result.loc[unfilled] += w * self.signals.loc[unfilled, s]

        return result

    # ── Disagreement detection ──────────────────────────────────────────

    def _detect_disagreements(self) -> List[DisagreementEvent]:
        """Detect periods where strategies conflict."""
        events: List[DisagreementEvent] = []
        signs = np.sign(self.signals)

        for i, date in enumerate(self.signals.index):
            row = signs.iloc[i]
            nonzero = row[row != 0]
            if len(nonzero) < 2:
                continue

            # Agreement: fraction of strategies agreeing with majority
            majority = np.sign(nonzero.sum())
            if majority == 0:
                agreement = 0.0
            else:
                agreement = float((nonzero == majority).mean())

            if agreement < (1.0 - self.disagreement_threshold):
                sizing = max(self.min_confidence, agreement)
                sig_dict = {s: float(self.signals.loc[date, s]) for s in self.strategies}
                events.append(DisagreementEvent(
                    date=str(date),
                    signals=sig_dict,
                    agreement_score=agreement,
                    recommended_sizing=sizing,
                    regime=str(self.regimes.get(date, "neutral")),
                ))
        return events

    # ── Ensemble confidence ─────────────────────────────────────────────

    def _ensemble_confidence(self) -> EnsembleConfidence:
        """Compute aggregate ensemble confidence metrics."""
        if self.ensemble_signals is None or len(self.ensemble_signals) == 0:
            return EnsembleConfidence(0.0, 0.0, 0.0, 0.0, 0.0)

        # Agreement ratio: how often strategies agree with ensemble direction
        ens_sign = np.sign(self.ensemble_signals)
        strat_signs = np.sign(self.signals)
        agreements = []
        for s in self.strategies:
            mask = ens_sign != 0
            if mask.sum() > 0:
                agreements.append(float((strat_signs.loc[mask, s] == ens_sign[mask]).mean()))
        agreement_ratio = float(np.mean(agreements)) if agreements else 0.0

        # Signal dispersion
        dispersion = float(self.signals.std(axis=1).mean())

        # Weight concentration (HHI)
        w_vals = np.array([self.current_weights[s].weight for s in self.strategies])
        hhi = float(np.sum(w_vals ** 2))

        # Regime stability: std of weight across regimes
        if len(self.regime_weights) >= 2:
            all_w = []
            for rw in self.regime_weights.values():
                all_w.append([rw.weights[s] for s in self.strategies])
            regime_std = float(np.mean(np.std(all_w, axis=0)))
            stability = max(0.0, 1.0 - regime_std * 4)
        else:
            stability = 1.0

        # Composite confidence
        score = (
            0.35 * agreement_ratio
            + 0.25 * (1.0 - min(dispersion * 10, 1.0))
            + 0.20 * (1.0 - hhi)
            + 0.20 * stability
        )
        score = max(0.0, min(1.0, score))

        return EnsembleConfidence(
            score=score,
            agreement_ratio=agreement_ratio,
            signal_dispersion=dispersion,
            weight_concentration=hhi,
            regime_stability=stability,
        )

    # ── Performance attribution ─────────────────────────────────────────

    def _performance_attribution(self) -> List[PerformanceAttribution]:
        """Attribute ensemble performance to individual strategies."""
        results: List[PerformanceAttribution] = []
        for s in self.strategies:
            strat_ret = self.signals[s] * self.returns
            total = float(strat_ret.sum())
            w = self.current_weights[s].weight
            contribution = w * total
            hits = strat_ret > 0
            hit_rate = float(hits.mean()) if len(hits) > 0 else 0.0
            avg_strength = float(self.signals[s].abs().mean())

            if self.ensemble_signals is not None and len(self.ensemble_signals) > 2:
                corr = float(self.signals[s].corr(self.ensemble_signals))
            else:
                corr = 0.0

            results.append(PerformanceAttribution(
                strategy=s, total_return=total,
                contribution=contribution, hit_rate=hit_rate,
                avg_signal_strength=avg_strength,
                correlation_with_ensemble=corr,
            ))
        return sorted(results, key=lambda a: -a.contribution)

    # ── Rolling weight evolution ────────────────────────────────────────

    def _rolling_weight_evolution(self) -> List[WeightSnapshot]:
        """Compute weight snapshots over time for the evolution chart."""
        snapshots: List[WeightSnapshot] = []
        step = max(1, self.lookback // 4)
        if len(self.returns) < self.lookback:
            return snapshots

        for end in range(self.lookback, len(self.returns), step):
            start = end - self.lookback
            window_ret = self.returns.iloc[start:end]
            window_sig = self.signals.iloc[start:end]
            exp_w = self._exponential_weights(len(window_ret))
            exp_w.index = window_ret.index

            raw: Dict[str, float] = {}
            for s in self.strategies:
                recent = window_sig[s] * window_ret
                sharpe = self._sharpe(recent)
                wr = float((recent > 0).mean())
                raw[s] = max(sharpe, 0.0) + 0.5 * wr
            total = sum(raw.values())
            weights = {s: raw[s] / total if total > 0 else 1.0 / len(self.strategies) for s in self.strategies}

            date = self.returns.index[end - 1]
            ens_sig = sum(weights[s] * float(self.signals.loc[date, s]) for s in self.strategies)

            snapshots.append(WeightSnapshot(
                date=str(date), weights=weights,
                ensemble_signal=ens_sig,
                regime=str(self.regimes.get(date, "neutral")),
            ))
        return snapshots

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: pd.Series, ann: float = 252.0) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(ann))

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report. Runs analyze() if not yet run."""
        if self.confidence is None:
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
        charts["weight_evolution"] = self._chart_weight_evolution()
        charts["agreement_heatmap"] = self._chart_agreement_heatmap()
        charts["attribution"] = self._chart_attribution()
        charts["regime_weights"] = self._chart_regime_weights()
        return charts

    def _chart_weight_evolution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.weight_history:
            return ""
        fig, ax = plt.subplots(figsize=(10, 4))
        xs = list(range(len(self.weight_history)))
        for s in self.strategies:
            ys = [snap.weights.get(s, 0) for snap in self.weight_history]
            ax.plot(xs, ys, label=s, lw=1.2)
        ax.set_ylabel("Weight")
        ax.set_title("Strategy Weight Evolution", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        # Regime background
        regime_colors = {
            "bull": "#16a34a", "bear": "#dc2626",
            "high_vol": "#f59e0b", "neutral": "#64748b",
        }
        for k in range(len(self.weight_history) - 1):
            c = regime_colors.get(self.weight_history[k].regime, "#f8fafc")
            ax.axvspan(k, k + 1, alpha=0.06, color=c)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_agreement_heatmap(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(self.strategies)
        agreement_matrix = np.zeros((n, n))
        signs = np.sign(self.signals)
        for i in range(n):
            for j in range(n):
                mask = (signs.iloc[:, i] != 0) & (signs.iloc[:, j] != 0)
                if mask.sum() > 0:
                    agreement_matrix[i, j] = float(
                        (signs.iloc[:, i][mask] == signs.iloc[:, j][mask]).mean()
                    )

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(agreement_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(self.strategies, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(self.strategies, fontsize=8)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{agreement_matrix[i, j]:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if agreement_matrix[i, j] < 0.5 else "black")
        fig.colorbar(im, shrink=0.8)
        ax.set_title("Strategy Agreement Heatmap", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_attribution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.attributions:
            return ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        names = [a.strategy for a in self.attributions]
        contribs = [a.contribution for a in self.attributions]
        colors = ["#16a34a" if c >= 0 else "#dc2626" for c in contribs]
        ax1.barh(names, contribs, color=colors, alpha=0.85)
        ax1.set_xlabel("Weighted Contribution")
        ax1.set_title("Performance Attribution", fontsize=10)
        ax1.grid(True, axis="x", alpha=0.3)

        hit_rates = [a.hit_rate for a in self.attributions]
        ax2.barh(names, hit_rates, color="#3b82f6", alpha=0.85)
        ax2.set_xlabel("Hit Rate")
        ax2.set_title("Strategy Hit Rates", fontsize=10)
        ax2.set_xlim(0, 1)
        ax2.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_regime_weights(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.regime_weights:
            return ""
        regimes = [r for r in REGIMES if r in self.regime_weights]
        if not regimes:
            return ""

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(regimes))
        width = 0.8 / len(self.strategies)
        colors = ["#3b82f6", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6", "#06b6d4"]

        for k, s in enumerate(self.strategies):
            vals = [self.regime_weights[r].weights.get(s, 0) for r in regimes]
            ax.bar(x + k * width, vals, width, label=s,
                   color=colors[k % len(colors)], alpha=0.85)
        ax.set_xticks(x + width * (len(self.strategies) - 1) / 2)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.set_ylabel("Weight")
        ax.set_title("Regime-Conditional Weights", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        conf = self.confidence or EnsembleConfidence(0, 0, 0, 0, 0)

        conf_color = "#16a34a" if conf.score >= 0.6 else "#f59e0b" if conf.score >= 0.4 else "#dc2626"

        # Current weights table
        weight_rows = ""
        for s in sorted(self.current_weights.values(), key=lambda w: -w.weight):
            weight_rows += (
                f'<tr><td>{s.strategy}</td>'
                f'<td>{s.weight:.1%}</td>'
                f'<td>{s.recent_sharpe:.2f}</td>'
                f'<td>{s.recent_win_rate:.1%}</td>'
                f'<td>{s.n_trades}</td></tr>\n'
            )

        # Regime weights table
        regime_rows = ""
        for regime in REGIMES:
            rw = self.regime_weights.get(regime)
            if rw is None:
                continue
            w_str = " / ".join(f"{rw.weights[s]:.0%}" for s in self.strategies)
            regime_rows += (
                f'<tr><td>{rw.regime}</td><td>{rw.n_obs}</td>'
                f'<td>{w_str}</td>'
                f'<td>{rw.ensemble_sharpe:.2f}</td></tr>\n'
            )

        # Attribution table
        attr_rows = ""
        for a in self.attributions:
            cls = "good" if a.contribution >= 0 else "bad"
            attr_rows += (
                f'<tr><td>{a.strategy}</td>'
                f'<td class="{cls}">{a.total_return:.2f}</td>'
                f'<td class="{cls}">{a.contribution:.2f}</td>'
                f'<td>{a.hit_rate:.1%}</td>'
                f'<td>{a.avg_signal_strength:.3f}</td>'
                f'<td>{a.correlation_with_ensemble:.2f}</td></tr>\n'
            )

        # Disagreement table
        disagree_rows = ""
        for d in self.disagreements[:20]:
            sig_str = " / ".join(f"{d.signals[s]:+.2f}" for s in self.strategies)
            disagree_rows += (
                f'<tr><td>{d.date}</td>'
                f'<td>{sig_str}</td>'
                f'<td class="{"bad" if d.agreement_score < 0.5 else ""}">{d.agreement_score:.1%}</td>'
                f'<td>{d.recommended_sizing:.0%}</td>'
                f'<td>{d.regime}</td></tr>\n'
            )
        if not disagree_rows:
            disagree_rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">No disagreements detected</td></tr>'

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        strat_header = " / ".join(self.strategies)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Strategy Ensemble Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
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

<h1>Strategy Ensemble Analysis</h1>
<div class="meta">{len(self.strategies)} strategies &middot; {len(self.returns)} observations &middot; Method: {self.method} &middot; Lookback: {self.lookback}d &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value" style="color:{conf_color}">{conf.score:.0%}</div><div class="label">Ensemble Confidence</div></div>
  <div class="kpi"><div class="value">{conf.agreement_ratio:.0%}</div><div class="label">Agreement Ratio</div></div>
  <div class="kpi"><div class="value">{len(self.disagreements)}</div><div class="label">Disagreement Events</div></div>
  <div class="kpi"><div class="value">{conf.weight_concentration:.2f}</div><div class="label">Weight HHI</div></div>
  <div class="kpi"><div class="value">{conf.regime_stability:.0%}</div><div class="label">Regime Stability</div></div>
</div>

<h2>1. Current Strategy Weights</h2>
<table>
<thead><tr><th>Strategy</th><th>Weight</th><th>Recent Sharpe</th><th>Win Rate</th><th>Trades</th></tr></thead>
<tbody>{weight_rows}</tbody>
</table>

<h2>2. Weight Evolution</h2>
{_img("weight_evolution")}

<h2>3. Strategy Agreement</h2>
{_img("agreement_heatmap")}

<h2>4. Regime-Conditional Weights</h2>
{_img("regime_weights")}
<table>
<thead><tr><th>Regime</th><th>Obs</th><th>Weights ({strat_header})</th><th>Ensemble Sharpe</th></tr></thead>
<tbody>{regime_rows}</tbody>
</table>

<h2>5. Performance Attribution</h2>
{_img("attribution")}
<table>
<thead><tr><th>Strategy</th><th>Total Return</th><th>Contribution</th><th>Hit Rate</th><th>Avg Signal</th><th>Corr w/ Ensemble</th></tr></thead>
<tbody>{attr_rows}</tbody>
</table>

<h2>6. Disagreement Detection</h2>
<p>Periods where strategy signals conflict (agreement &lt; {1.0 - self.disagreement_threshold:.0%}). Position size reduced proportionally.</p>
<table>
<thead><tr><th>Date</th><th>Signals ({strat_header})</th><th>Agreement</th><th>Sizing</th><th>Regime</th></tr></thead>
<tbody>{disagree_rows}</tbody>
</table>

<footer>Generated by <code>compass/strategy_ensemble.py</code></footer>
</body></html>"""
        return html
