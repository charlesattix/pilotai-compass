"""
Real-time signal quality assessment across experiments.

Track per-signal: IC, ICIR, hit rate, profit factor, win/loss ratio,
Sharpe, turnover, decay rate. Rolling quality score (0-100), automatic
grading (A-F), stale signal detection, cross-signal correlation.

Usage::

    from compass.signal_quality_scorer import SignalQualityScorer
    scorer = SignalQualityScorer(signals_df, returns)
    results = scorer.analyze()
    scorer.generate_report()
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
DEFAULT_OUTPUT = ROOT / "reports" / "signal_quality.html"


@dataclass
class SignalMetrics:
    name: str
    ic: float                # information coefficient (rank corr with fwd returns)
    icir: float              # IC / std(IC)
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    sharpe: float
    turnover: float          # fraction of signal that changes per period
    decay_rate: float        # half-life of IC in periods
    quality_score: float     # 0-100 composite
    grade: str               # A-F
    is_stale: bool


@dataclass
class SignalCorrelation:
    signal_a: str
    signal_b: str
    correlation: float


@dataclass
class QualitySummary:
    n_signals: int
    avg_score: float
    n_stale: int
    grade_distribution: Dict[str, int]
    best_signal: str
    worst_signal: str


class SignalQualityScorer:
    """Score signal quality across multiple experiments."""

    def __init__(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        window: int = 60,
        stale_threshold: int = 10,
        forward_periods: int = 5,
    ) -> None:
        self.signals = signals.copy()
        self.returns = returns.copy()
        self.signal_names = list(signals.columns)
        self.window = window
        self.stale_threshold = stale_threshold
        self.forward_periods = forward_periods

        common = signals.index.intersection(returns.index)
        self.signals = self.signals.loc[common]
        self.returns = self.returns.loc[common]

        self.metrics: List[SignalMetrics] = []
        self.correlations: List[SignalCorrelation] = []
        self.summary: Optional[QualitySummary] = None

    @classmethod
    def from_csv(cls, sig_path: str, ret_path: str, **kwargs) -> "SignalQualityScorer":
        sig = pd.read_csv(sig_path, index_col=0, parse_dates=True)
        ret = pd.read_csv(ret_path, index_col=0, parse_dates=True).squeeze("columns")
        return cls(sig, ret, **kwargs)

    def analyze(self) -> Dict[str, Any]:
        self.metrics = [self._score_signal(name) for name in self.signal_names]
        self.correlations = self._cross_correlations()
        self.summary = self._summarize()
        return {"metrics": self.metrics, "correlations": self.correlations, "summary": self.summary}

    def _score_signal(self, name: str) -> SignalMetrics:
        sig = self.signals[name]
        fwd = self.returns.shift(-self.forward_periods)
        mask = sig.notna() & fwd.notna()
        s, f = sig[mask], fwd[mask]

        if len(s) < 10:
            return SignalMetrics(name, 0, 0, 0.5, 1, 0, 0, 1, 0, 0, 0, 0, "F", True)

        # IC: rank correlation
        ic = float(s.corr(f, method="spearman"))

        # Rolling IC for ICIR and decay
        rolling_ic = s.rolling(self.window).corr(f).dropna()
        icir = float(rolling_ic.mean() / rolling_ic.std()) if len(rolling_ic) > 1 and rolling_ic.std() > 0 else 0

        # Strategy returns
        strat_ret = sig * self.returns
        strat_ret = strat_ret.dropna()

        wins = strat_ret[strat_ret > 0]
        losses = strat_ret[strat_ret < 0]
        hit_rate = float(len(wins) / len(strat_ret)) if len(strat_ret) > 0 else 0.5
        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 1.0
        gross_wins = float(wins.sum()) if len(wins) > 0 else 0
        gross_losses = abs(float(losses.sum())) if len(losses) > 0 else 0
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 1.0

        sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0

        # Turnover
        changes = (sig.diff().abs() > 0).sum()
        turnover = float(changes / max(len(sig) - 1, 1))

        # Decay rate: half-life of rolling IC autocorrelation
        if len(rolling_ic) > 20:
            ac = float(rolling_ic.autocorr())
            decay = -1 / np.log(abs(ac)) if 0 < abs(ac) < 1 else 0
        else:
            decay = 0

        # Stale detection
        is_stale = int((sig.diff() == 0).sum()) > len(sig) * 0.8

        # Composite score
        score = self._compute_score(ic, icir, hit_rate, pf, sharpe, turnover, is_stale)
        grade = self._grade(score)

        return SignalMetrics(
            name=name, ic=ic, icir=icir, hit_rate=hit_rate,
            profit_factor=min(pf, 99), avg_win=avg_win, avg_loss=avg_loss,
            win_loss_ratio=min(wl_ratio, 99), sharpe=sharpe,
            turnover=turnover, decay_rate=decay,
            quality_score=score, grade=grade, is_stale=is_stale,
        )

    @staticmethod
    def _compute_score(ic, icir, hit_rate, pf, sharpe, turnover, stale) -> float:
        if stale:
            return 0.0
        s = 0.0
        s += min(20, abs(ic) * 200)              # IC: up to 20
        s += min(15, abs(icir) * 15)              # ICIR: up to 15
        s += min(20, (hit_rate - 0.4) * 100)      # Hit rate: up to 20
        s += min(15, min(pf, 5) * 3)              # Profit factor: up to 15
        s += min(20, max(sharpe, 0) * 5)           # Sharpe: up to 20
        s += min(10, (1 - turnover) * 10)          # Low turnover: up to 10
        return max(0, min(100, s))

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 80: return "A"
        if score >= 65: return "B"
        if score >= 50: return "C"
        if score >= 35: return "D"
        return "F"

    def _cross_correlations(self) -> List[SignalCorrelation]:
        results = []
        n = len(self.signal_names)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.signal_names[i], self.signal_names[j]
                corr = float(self.signals[a].corr(self.signals[b]))
                results.append(SignalCorrelation(a, b, corr))
        return sorted(results, key=lambda c: -abs(c.correlation))

    def _summarize(self) -> QualitySummary:
        if not self.metrics:
            return QualitySummary(0, 0, 0, {}, "", "")
        scores = [m.quality_score for m in self.metrics]
        grades: Dict[str, int] = {}
        for m in self.metrics:
            grades[m.grade] = grades.get(m.grade, 0) + 1
        best = max(self.metrics, key=lambda m: m.quality_score)
        worst = min(self.metrics, key=lambda m: m.quality_score)
        return QualitySummary(
            len(self.metrics), float(np.mean(scores)),
            sum(1 for m in self.metrics if m.is_stale),
            grades, best.name, worst.name,
        )

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.summary is None:
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
        return {"scorecard": self._chart_scores(), "corr": self._chart_corr(), "grades": self._chart_grades()}

    def _chart_scores(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.metrics: return ""
        names = [m.name for m in sorted(self.metrics, key=lambda x: -x.quality_score)]
        scores = [m.quality_score for m in sorted(self.metrics, key=lambda x: -x.quality_score)]
        colors = ["#16a34a" if s >= 65 else "#f59e0b" if s >= 35 else "#dc2626" for s in scores]
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))
        ax.barh(names, scores, color=colors, alpha=0.85)
        ax.set_xlim(0, 100); ax.set_xlabel("Quality Score")
        ax.set_title("Signal Quality Scorecard", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_corr(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(self.signal_names)
        if n < 2: return ""
        matrix = np.eye(n)
        idx = {name: i for i, name in enumerate(self.signal_names)}
        for c in self.correlations:
            matrix[idx[c.signal_a], idx[c.signal_b]] = c.correlation
            matrix[idx[c.signal_b], idx[c.signal_a]] = c.correlation
        fig, ax = plt.subplots(figsize=(max(4, n * 0.7), max(4, n * 0.6)))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(self.signal_names, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(self.signal_names, fontsize=7)
        fig.colorbar(im, shrink=0.8); ax.set_title("Signal Correlation", fontsize=11); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_grades(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.summary: return ""
        d = self.summary.grade_distribution
        grades = ["A", "B", "C", "D", "F"]
        vals = [d.get(g, 0) for g in grades]
        colors = ["#16a34a", "#22c55e", "#f59e0b", "#f97316", "#dc2626"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(grades, vals, color=colors, alpha=0.85)
        ax.set_ylabel("Signals"); ax.set_title("Grade Distribution", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = self.summary or QualitySummary(0, 0, 0, {}, "", "")

        rows = ""
        for m in sorted(self.metrics, key=lambda x: -x.quality_score):
            cls = "good" if m.grade in ("A", "B") else "bad" if m.grade in ("D", "F") else ""
            stale = '<span class="bad">STALE</span>' if m.is_stale else ""
            rows += (f'<tr><td>{m.name} {stale}</td><td>{m.ic:+.3f}</td><td>{m.icir:.2f}</td>'
                    f'<td>{m.hit_rate:.0%}</td><td>{m.profit_factor:.1f}</td>'
                    f'<td>{m.sharpe:.2f}</td><td>{m.turnover:.0%}</td>'
                    f'<td>{m.decay_rate:.1f}</td>'
                    f'<td class="{cls}">{m.grade} ({m.quality_score:.0f})</td></tr>\n')

        corr_rows = ""
        for c in self.correlations[:15]:
            cls = "bad" if abs(c.correlation) > 0.7 else ""
            corr_rows += f'<tr><td>{c.signal_a} / {c.signal_b}</td><td class="{cls}">{c.correlation:.3f}</td></tr>\n'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Signal Quality Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
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
<h1>Signal Quality Report</h1>
<div class="meta">{s.n_signals} signals &middot; Avg score: {s.avg_score:.0f} &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value">{s.n_signals}</div><div class="label">Signals</div></div>
  <div class="kpi"><div class="value">{s.avg_score:.0f}</div><div class="label">Avg Score</div></div>
  <div class="kpi"><div class="value bad">{s.n_stale}</div><div class="label">Stale</div></div>
  <div class="kpi"><div class="value good">{s.best_signal}</div><div class="label">Best</div></div>
</div>
<h2>1. Signal Scorecard</h2>{_img("scorecard")}
<table><thead><tr><th>Signal</th><th>IC</th><th>ICIR</th><th>Hit Rate</th><th>PF</th><th>Sharpe</th><th>Turnover</th><th>Decay</th><th>Grade</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>2. Grade Distribution</h2>{_img("grades")}
<h2>3. Cross-Signal Correlation</h2>{_img("corr")}
<table><thead><tr><th>Pair</th><th>Correlation</th></tr></thead><tbody>{corr_rows}</tbody></table>
<footer>Generated by <code>compass/signal_quality_scorer.py</code></footer>
</body></html>"""
        return html
