"""
Automated research pipeline for signal discovery and validation.

Generates signal hypotheses, tests them with walk-forward,
applies multiple hypothesis correction, controls FDR,
clusters similar signals, and maintains a research log.

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class SignalHypothesis:
    name: str
    feature_a: str
    feature_b: Optional[str] = None
    interaction: str = "ratio"     # ratio | diff | product
    lookback: int = 20


@dataclass
class TestResult:
    hypothesis: SignalHypothesis
    is_sharpe: float
    oos_sharpe: float
    t_stat: float
    p_value: float
    ic: float
    passed_raw: bool


@dataclass
class CorrectedResult:
    name: str
    raw_p: float
    corrected_p: float
    method: str                    # "bonferroni" | "bh"
    is_significant: bool


@dataclass
class SignalCluster:
    cluster_id: int
    members: List[str]
    representative: str
    avg_correlation: float


@dataclass
class ResearchLogEntry:
    timestamp: datetime
    hypothesis: str
    result: str
    sharpe: float
    p_value: float


@dataclass
class PipelineResult:
    n_hypotheses: int
    n_tested: int
    n_raw_significant: int
    n_after_correction: int
    n_after_clustering: int
    corrected_results: List[CorrectedResult]
    clusters: List[SignalCluster]
    log: List[ResearchLogEntry]
    top_signals: List[TestResult]


class ResearchPipeline:
    """Automated signal research pipeline.

    Args:
        significance_level: Alpha for significance tests.
        correction_method: "bonferroni" or "bh" (Benjamini-Hochberg).
        cluster_threshold: Correlation above which signals are clustered.
        n_walk_forward_folds: Folds for walk-forward testing.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        correction_method: str = "bh",
        cluster_threshold: float = 0.70,
        n_walk_forward_folds: int = 5,
    ) -> None:
        self.significance_level = significance_level
        self.correction_method = correction_method
        self.cluster_threshold = cluster_threshold
        self.n_walk_forward_folds = n_walk_forward_folds
        self._log: List[ResearchLogEntry] = []

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_hypotheses(
        feature_names: List[str],
        lookbacks: Optional[List[int]] = None,
        max_hypotheses: int = 100,
    ) -> List[SignalHypothesis]:
        """Generate signal hypotheses from feature combinations."""
        lookbacks = lookbacks or [5, 10, 20, 50]
        hyps: List[SignalHypothesis] = []

        # Single features
        for f in feature_names:
            for lb in lookbacks:
                hyps.append(SignalHypothesis(
                    name=f"{f}_mom_{lb}", feature_a=f, lookback=lb))

        # Pairwise interactions
        for a, b in combinations(feature_names, 2):
            for inter in ["ratio", "diff"]:
                hyps.append(SignalHypothesis(
                    name=f"{a}_{inter}_{b}", feature_a=a, feature_b=b,
                    interaction=inter, lookback=20))

        return hyps[:max_hypotheses]

    # ------------------------------------------------------------------
    # Signal construction from hypothesis
    # ------------------------------------------------------------------

    @staticmethod
    def build_signal(
        hypothesis: SignalHypothesis,
        features: pd.DataFrame,
    ) -> pd.Series:
        """Construct a signal series from a hypothesis definition."""
        if hypothesis.feature_a not in features.columns:
            return pd.Series(dtype=float)

        a = features[hypothesis.feature_a]
        if hypothesis.feature_b and hypothesis.feature_b in features.columns:
            b = features[hypothesis.feature_b]
            if hypothesis.interaction == "ratio":
                sig = (a / b.replace(0, np.nan)).dropna()
            elif hypothesis.interaction == "diff":
                sig = a - b
            else:
                sig = a * b
        else:
            sig = a.pct_change(hypothesis.lookback)

        # Normalise to [-1, 1]
        s = sig.dropna()
        if s.empty or s.std() < 1e-12:
            return pd.Series(0.0, index=features.index)
        return ((s - s.mean()) / s.std()).clip(-3, 3) / 3

    # ------------------------------------------------------------------
    # Walk-forward test
    # ------------------------------------------------------------------

    def walk_forward_test(
        self,
        signal: pd.Series,
        returns: pd.Series,
    ) -> Tuple[float, float, float, float]:
        """Walk-forward test. Returns (is_sharpe, oos_sharpe, t_stat, p_value)."""
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        n = len(aligned)
        if n < 50:
            return 0.0, 0.0, 0.0, 1.0

        fold_size = n // self.n_walk_forward_folds
        oos_sharpes: List[float] = []
        is_sharpes: List[float] = []

        for i in range(self.n_walk_forward_folds - 1):
            train = aligned.iloc[:fold_size * (i + 1)]
            test = aligned.iloc[fold_size * (i + 1):fold_size * (i + 2)]
            if len(test) < 5:
                continue

            # IS
            is_ret = np.sign(train["sig"].shift(1)) * train["ret"]
            is_ret = is_ret.dropna()
            is_mu = float(is_ret.mean())
            is_std = float(is_ret.std())
            is_sharpes.append(is_mu / is_std * np.sqrt(TRADING_DAYS) if is_std > 1e-12 else 0.0)

            # OOS
            oos_ret = np.sign(test["sig"].shift(1)) * test["ret"]
            oos_ret = oos_ret.dropna()
            oos_mu = float(oos_ret.mean())
            oos_std = float(oos_ret.std())
            oos_sharpes.append(oos_mu / oos_std * np.sqrt(TRADING_DAYS) if oos_std > 1e-12 else 0.0)

        if not oos_sharpes:
            return 0.0, 0.0, 0.0, 1.0

        avg_is = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))

        # t-test on OOS Sharpes
        oos_arr = np.array(oos_sharpes)
        if len(oos_arr) < 2 or oos_arr.std() < 1e-12:
            return avg_is, avg_oos, 0.0, 1.0
        t_stat = float(oos_arr.mean() / (oos_arr.std() / np.sqrt(len(oos_arr))))
        from scipy import stats as sp_stats
        p_value = float(1 - sp_stats.t.cdf(abs(t_stat), df=len(oos_arr) - 1))

        return avg_is, avg_oos, t_stat, p_value

    # ------------------------------------------------------------------
    # Multiple hypothesis correction
    # ------------------------------------------------------------------

    def correct_pvalues(
        self, results: List[TestResult],
    ) -> List[CorrectedResult]:
        """Apply Bonferroni or Benjamini-Hochberg correction."""
        n = len(results)
        if n == 0:
            return []

        raw_ps = [r.p_value for r in results]

        if self.correction_method == "bonferroni":
            corrected = [min(p * n, 1.0) for p in raw_ps]
        else:  # BH
            sorted_idx = np.argsort(raw_ps)
            corrected = [0.0] * n
            for rank, idx in enumerate(sorted_idx):
                corrected[idx] = min(raw_ps[idx] * n / (rank + 1), 1.0)
            # Enforce monotonicity
            for i in range(len(sorted_idx) - 2, -1, -1):
                idx = sorted_idx[i]
                next_idx = sorted_idx[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])

        output: List[CorrectedResult] = []
        for i, r in enumerate(results):
            output.append(CorrectedResult(
                name=r.hypothesis.name,
                raw_p=raw_ps[i], corrected_p=corrected[i],
                method=self.correction_method,
                is_significant=corrected[i] < self.significance_level,
            ))
        return output

    # ------------------------------------------------------------------
    # Signal clustering
    # ------------------------------------------------------------------

    def cluster_signals(
        self,
        signals: Dict[str, pd.Series],
        significant_names: List[str],
    ) -> List[SignalCluster]:
        """Cluster correlated signals, keep most parsimonious."""
        if len(significant_names) <= 1:
            return [SignalCluster(0, significant_names,
                                   significant_names[0] if significant_names else "", 0.0)]

        df = pd.DataFrame({n: signals[n] for n in significant_names if n in signals}).dropna()
        if df.empty or len(df.columns) < 2:
            return [SignalCluster(0, significant_names, significant_names[0], 0.0)]

        corr = df.corr().abs()
        assigned = set()
        clusters: List[SignalCluster] = []
        cid = 0

        for name in significant_names:
            if name in assigned or name not in corr.columns:
                continue
            members = [name]
            for other in significant_names:
                if other != name and other not in assigned and other in corr.columns:
                    if corr.loc[name, other] >= self.cluster_threshold:
                        members.append(other)
            assigned.update(members)
            avg_c = float(corr.loc[members, members].values[np.triu_indices(len(members), 1)].mean()) if len(members) > 1 else 0.0
            # Representative: shortest name (most parsimonious)
            rep = min(members, key=len)
            clusters.append(SignalCluster(cid, members, rep, avg_c))
            cid += 1

        return clusters

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        feature_names: Optional[List[str]] = None,
        max_hypotheses: int = 50,
    ) -> PipelineResult:
        """Run the full research pipeline."""
        if feature_names is None:
            feature_names = features.columns.tolist()

        hypotheses = self.generate_hypotheses(feature_names, max_hypotheses=max_hypotheses)

        test_results: List[TestResult] = []
        signals: Dict[str, pd.Series] = {}

        for h in hypotheses:
            sig = self.build_signal(h, features)
            if sig.empty or sig.std() < 1e-12:
                continue
            signals[h.name] = sig
            is_sh, oos_sh, t, p = self.walk_forward_test(sig, returns)
            ic = float(sig.corr(returns.shift(-1), method="spearman")) if len(sig) > 20 else 0.0
            passed = p < self.significance_level
            test_results.append(TestResult(
                hypothesis=h, is_sharpe=is_sh, oos_sharpe=oos_sh,
                t_stat=t, p_value=p, ic=ic, passed_raw=passed,
            ))

            self._log.append(ResearchLogEntry(
                timestamp=datetime.now(), hypothesis=h.name,
                result="pass" if passed else "fail",
                sharpe=oos_sh, p_value=p,
            ))

        corrected = self.correct_pvalues(test_results)
        sig_names = [c.name for c in corrected if c.is_significant]

        clusters = self.cluster_signals(signals, sig_names)
        reps = {c.representative for c in clusters}

        top = sorted(
            [r for r in test_results if r.hypothesis.name in reps],
            key=lambda r: r.oos_sharpe, reverse=True,
        )

        return PipelineResult(
            n_hypotheses=len(hypotheses), n_tested=len(test_results),
            n_raw_significant=sum(1 for r in test_results if r.passed_raw),
            n_after_correction=len(sig_names),
            n_after_clustering=len(reps),
            corrected_results=corrected,
            clusters=clusters,
            log=self._log,
            top_signals=top,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, result: PipelineResult,
        output_path: str = "reports/research_pipeline.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        top_rows = [
            f"<tr><td>{r.hypothesis.name}</td><td>{r.oos_sharpe:.2f}</td>"
            f"<td>{r.p_value:.4f}</td><td>{r.ic:.4f}</td></tr>"
            for r in result.top_signals[:10]
        ]
        cluster_rows = [
            f"<tr><td>{c.cluster_id}</td><td>{c.representative}</td>"
            f"<td>{len(c.members)}</td><td>{c.avg_correlation:.2f}</td></tr>"
            for c in result.clusters
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Research Pipeline</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Research Pipeline Report</h1>
<div class="summary">
<p>Hypotheses: {result.n_hypotheses} | Tested: {result.n_tested} |
   Raw Significant: {result.n_raw_significant} |
   After Correction: {result.n_after_correction} |
   After Clustering: {result.n_after_clustering}</p>
</div>
<h2>Top Signals</h2>
<table><tr><th>Signal</th><th>OOS Sharpe</th><th>p-value</th><th>IC</th></tr>
{''.join(top_rows)}</table>
<h2>Signal Clusters</h2>
<table><tr><th>ID</th><th>Representative</th><th>Members</th><th>Avg Corr</th></tr>
{''.join(cluster_rows)}</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
