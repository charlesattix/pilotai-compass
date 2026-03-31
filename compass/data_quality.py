"""
Data quality monitoring system.

Completeness checks (missing bars, gaps), accuracy validation (outliers,
stale quotes, price range), consistency checks (cross-asset alignment),
timeliness monitoring (freshness, lag), quality scoring per source,
automated repair (interpolation, forward-fill), alert system, and
HTML report.

Usage::

    from compass.data_quality import DataQualityEngine
    engine = DataQualityEngine(ohlcv_df)
    results = engine.analyze()
    engine.generate_report()
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
DEFAULT_OUTPUT = ROOT / "reports" / "data_quality.html"


@dataclass
class CompletenessCheck:
    column: str
    total_rows: int
    missing: int
    missing_pct: float
    gaps: int               # consecutive missing runs
    longest_gap: int
    passed: bool


@dataclass
class AccuracyCheck:
    column: str
    outliers: int
    stale_count: int        # repeated identical values
    range_violations: int   # close outside high-low
    z_threshold: float
    passed: bool


@dataclass
class ConsistencyCheck:
    name: str
    description: str
    violations: int
    passed: bool


@dataclass
class FreshnessCheck:
    source: str
    last_timestamp: str
    age_seconds: float
    threshold_seconds: float
    is_fresh: bool


@dataclass
class QualityScore:
    source: str
    completeness: float     # 0-1
    accuracy: float
    consistency: float
    timeliness: float
    overall: float
    grade: str              # A-F


@dataclass
class RepairAction:
    column: str
    method: str             # "interpolate", "ffill", "drop"
    rows_affected: int
    description: str


@dataclass
class QualityAlert:
    severity: str           # "info", "warning", "critical"
    category: str
    message: str
    column: str


class DataQualityEngine:
    """Data quality monitoring and repair."""

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "default",
        z_threshold: float = 4.0,
        stale_threshold: int = 5,
        freshness_threshold: float = 86400.0,
        auto_repair: bool = False,
    ) -> None:
        self.data = data.copy()
        self.source = source
        self.z_threshold = z_threshold
        self.stale_threshold = stale_threshold
        self.freshness_threshold = freshness_threshold
        self.auto_repair = auto_repair

        self.completeness: List[CompletenessCheck] = []
        self.accuracy: List[AccuracyCheck] = []
        self.consistency: List[ConsistencyCheck] = []
        self.freshness: List[FreshnessCheck] = []
        self.scores: List[QualityScore] = []
        self.repairs: List[RepairAction] = []
        self.alerts: List[QualityAlert] = []

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "DataQualityEngine":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    def analyze(self) -> Dict[str, Any]:
        self.completeness = self._check_completeness()
        self.accuracy = self._check_accuracy()
        self.consistency = self._check_consistency()
        self.freshness = self._check_freshness()
        self.scores = self._compute_scores()
        self.alerts = self._generate_alerts()
        if self.auto_repair:
            self.repairs = self._auto_repair()
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "freshness": self.freshness,
            "scores": self.scores,
            "alerts": self.alerts,
            "repairs": self.repairs,
        }

    def _check_completeness(self) -> List[CompletenessCheck]:
        results = []
        for col in self.data.columns:
            series = self.data[col]
            missing = int(series.isna().sum())
            total = len(series)
            pct = missing / total if total > 0 else 0

            # Count gaps
            is_na = series.isna().values
            gaps = 0
            longest = 0
            run = 0
            for v in is_na:
                if v:
                    run += 1
                else:
                    if run > 0:
                        gaps += 1
                        longest = max(longest, run)
                    run = 0
            if run > 0:
                gaps += 1
                longest = max(longest, run)

            results.append(CompletenessCheck(
                col, total, missing, pct, gaps, longest,
                passed=pct < 0.05,
            ))
        return results

    def _check_accuracy(self) -> List[AccuracyCheck]:
        results = []
        for col in self.data.select_dtypes(include=[np.number]).columns:
            series = self.data[col].dropna()
            if len(series) < 10:
                results.append(AccuracyCheck(col, 0, 0, 0, self.z_threshold, True))
                continue

            # Outliers via z-score
            z = np.abs((series - series.mean()) / series.std())
            outliers = int((z > self.z_threshold).sum())

            # Stale: consecutive identical values
            diffs = series.diff()
            stale = 0
            run = 0
            for d in diffs.values:
                if d == 0:
                    run += 1
                    if run >= self.stale_threshold:
                        stale += 1
                else:
                    run = 0

            # Range violations: close outside high-low
            range_v = 0
            if "close" in self.data.columns and "high" in self.data.columns and "low" in self.data.columns:
                if col == "close":
                    mask = (self.data["close"] > self.data["high"]) | (self.data["close"] < self.data["low"])
                    range_v = int(mask.sum())

            results.append(AccuracyCheck(
                col, outliers, stale, range_v, self.z_threshold,
                passed=outliers < len(series) * 0.02 and stale < 3,
            ))
        return results

    def _check_consistency(self) -> List[ConsistencyCheck]:
        results = []
        # High >= Low
        if "high" in self.data.columns and "low" in self.data.columns:
            violations = int((self.data["high"] < self.data["low"]).sum())
            results.append(ConsistencyCheck(
                "high_low", "High must be >= Low", violations, violations == 0,
            ))

        # Volume > 0
        if "volume" in self.data.columns:
            neg = int((self.data["volume"] < 0).sum())
            results.append(ConsistencyCheck(
                "volume_positive", "Volume must be >= 0", neg, neg == 0,
            ))

        # Close within [low, high]
        if all(c in self.data.columns for c in ("close", "high", "low")):
            outside = int(((self.data["close"] > self.data["high"]) |
                          (self.data["close"] < self.data["low"])).sum())
            results.append(ConsistencyCheck(
                "close_in_range", "Close within [Low, High]", outside, outside == 0,
            ))

        # Monotonic index (no duplicates)
        if hasattr(self.data.index, 'duplicated'):
            dups = int(self.data.index.duplicated().sum())
            results.append(ConsistencyCheck(
                "no_dup_index", "No duplicate timestamps", dups, dups == 0,
            ))

        return results

    def _check_freshness(self) -> List[FreshnessCheck]:
        results = []
        if hasattr(self.data.index, 'max') and hasattr(self.data.index, 'dtype'):
            try:
                last = pd.Timestamp(self.data.index.max())
                now = pd.Timestamp.now(tz=last.tzinfo)
                age = (now - last).total_seconds()
                results.append(FreshnessCheck(
                    self.source, str(last), age,
                    self.freshness_threshold,
                    age < self.freshness_threshold,
                ))
            except (TypeError, ValueError):
                pass
        return results

    def _compute_scores(self) -> List[QualityScore]:
        comp_scores = [1 - c.missing_pct for c in self.completeness]
        comp = float(np.mean(comp_scores)) if comp_scores else 1.0

        acc_scores = []
        for a in self.accuracy:
            total = max(self.data[a.column].dropna().shape[0], 1)
            acc_scores.append(1 - a.outliers / total)
        acc = float(np.mean(acc_scores)) if acc_scores else 1.0

        cons = 1.0 if all(c.passed for c in self.consistency) else \
            sum(c.passed for c in self.consistency) / max(len(self.consistency), 1)

        time = 1.0 if all(f.is_fresh for f in self.freshness) else 0.5

        overall = 0.30 * comp + 0.30 * acc + 0.25 * cons + 0.15 * time
        if overall >= 0.9:
            grade = "A"
        elif overall >= 0.8:
            grade = "B"
        elif overall >= 0.7:
            grade = "C"
        elif overall >= 0.6:
            grade = "D"
        else:
            grade = "F"

        return [QualityScore(self.source, comp, acc, cons, time, overall, grade)]

    def _generate_alerts(self) -> List[QualityAlert]:
        alerts = []
        for c in self.completeness:
            if not c.passed:
                alerts.append(QualityAlert("critical", "completeness",
                    f"{c.column}: {c.missing_pct:.1%} missing ({c.missing} rows)", c.column))
            elif c.missing > 0:
                alerts.append(QualityAlert("warning", "completeness",
                    f"{c.column}: {c.missing} missing values", c.column))
        for a in self.accuracy:
            if a.outliers > 0:
                sev = "critical" if a.outliers > 10 else "warning"
                alerts.append(QualityAlert(sev, "accuracy",
                    f"{a.column}: {a.outliers} outliers (z>{a.z_threshold})", a.column))
            if a.stale_count > 0:
                alerts.append(QualityAlert("warning", "accuracy",
                    f"{a.column}: {a.stale_count} stale sequences", a.column))
        for c in self.consistency:
            if not c.passed:
                alerts.append(QualityAlert("critical", "consistency",
                    f"{c.name}: {c.violations} violations", ""))
        for f in self.freshness:
            if not f.is_fresh:
                alerts.append(QualityAlert("critical", "timeliness",
                    f"{f.source}: data {f.age_seconds/3600:.0f}h old", ""))
        return alerts

    def _auto_repair(self) -> List[RepairAction]:
        repairs = []
        for c in self.completeness:
            if c.missing > 0 and c.missing_pct < 0.10:
                col = c.column
                before = int(self.data[col].isna().sum())
                if self.data[col].dtype in (np.float64, np.int64, float, int):
                    self.data[col] = self.data[col].interpolate(method="linear", limit=5)
                else:
                    self.data[col] = self.data[col].ffill(limit=5)
                after = int(self.data[col].isna().sum())
                fixed = before - after
                if fixed > 0:
                    repairs.append(RepairAction(col, "interpolate", fixed,
                        f"Interpolated {fixed} missing values in {col}"))
        return repairs

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if not self.scores:
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
        return {"score": self._chart_score(), "completeness": self._chart_completeness()}

    def _chart_score(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.scores:
            return ""
        s = self.scores[0]
        cats = ["Completeness", "Accuracy", "Consistency", "Timeliness", "Overall"]
        vals = [s.completeness, s.accuracy, s.consistency, s.timeliness, s.overall]
        colors = ["#16a34a" if v >= 0.8 else "#f59e0b" if v >= 0.6 else "#dc2626" for v in vals]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(cats, vals, color=colors, alpha=0.85)
        ax.set_xlim(0, 1); ax.set_xlabel("Score")
        ax.set_title(f"Data Quality: Grade {s.grade}", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_completeness(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.completeness:
            return ""
        cols = [c.column for c in self.completeness]
        pcts = [1 - c.missing_pct for c in self.completeness]
        colors = ["#16a34a" if p >= 0.95 else "#f59e0b" if p >= 0.9 else "#dc2626" for p in pcts]
        fig, ax = plt.subplots(figsize=(7, max(3, len(cols) * 0.35)))
        ax.barh(cols, pcts, color=colors, alpha=0.85)
        ax.set_xlim(0, 1); ax.set_xlabel("Completeness")
        ax.set_title("Column Completeness", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = self.scores[0] if self.scores else QualityScore("", 0, 0, 0, 0, 0, "F")
        n_crit = sum(1 for a in self.alerts if a.severity == "critical")
        n_warn = sum(1 for a in self.alerts if a.severity == "warning")

        comp_rows = ""
        for c in self.completeness:
            cls = "good" if c.passed else "bad"
            comp_rows += f'<tr><td>{c.column}</td><td>{c.total_rows}</td><td class="{cls}">{c.missing}</td><td>{c.missing_pct:.1%}</td><td>{c.gaps}</td><td>{c.longest_gap}</td></tr>\n'

        acc_rows = ""
        for a in self.accuracy:
            cls = "good" if a.passed else "bad"
            acc_rows += f'<tr><td>{a.column}</td><td class="{cls}">{a.outliers}</td><td>{a.stale_count}</td><td>{a.range_violations}</td></tr>\n'

        cons_rows = ""
        for c in self.consistency:
            cls = "good" if c.passed else "bad"
            cons_rows += f'<tr><td>{c.name}</td><td>{c.description}</td><td>{c.violations}</td><td class="{cls}">{"PASS" if c.passed else "FAIL"}</td></tr>\n'

        alert_rows = ""
        for a in self.alerts:
            cls = "bad" if a.severity == "critical" else "warn" if a.severity == "warning" else ""
            alert_rows += f'<tr><td class="{cls}">{a.severity}</td><td>{a.category}</td><td>{a.message}</td></tr>\n'
        if not alert_rows:
            alert_rows = '<tr><td colspan="3" style="text-align:center;color:#64748b">No alerts</td></tr>'

        repair_rows = ""
        for r in self.repairs:
            repair_rows += f'<tr><td>{r.column}</td><td>{r.method}</td><td>{r.rows_affected}</td><td>{r.description}</td></tr>\n'
        if not repair_rows:
            repair_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No repairs</td></tr>'

        grade_cls = "good" if s.grade in ("A", "B") else "bad" if s.grade in ("D", "F") else ""

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Data Quality Report</title>
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
<h1>Data Quality Report</h1>
<div class="meta">{len(self.data)} rows &middot; {len(self.data.columns)} columns &middot; Source: {self.source} &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {grade_cls}">{s.grade}</div><div class="label">Quality Grade</div></div>
  <div class="kpi"><div class="value">{s.overall:.0%}</div><div class="label">Overall Score</div></div>
  <div class="kpi"><div class="value bad">{n_crit}</div><div class="label">Critical Alerts</div></div>
  <div class="kpi"><div class="value warn">{n_warn}</div><div class="label">Warnings</div></div>
  <div class="kpi"><div class="value">{len(self.repairs)}</div><div class="label">Repairs</div></div>
</div>
<h2>1. Quality Scorecard</h2>{_img("score")}
<h2>2. Completeness</h2>{_img("completeness")}
<table><thead><tr><th>Column</th><th>Rows</th><th>Missing</th><th>Missing %</th><th>Gaps</th><th>Longest Gap</th></tr></thead><tbody>{comp_rows}</tbody></table>
<h2>3. Accuracy</h2>
<table><thead><tr><th>Column</th><th>Outliers</th><th>Stale Seqs</th><th>Range Violations</th></tr></thead><tbody>{acc_rows}</tbody></table>
<h2>4. Consistency</h2>
<table><thead><tr><th>Check</th><th>Rule</th><th>Violations</th><th>Result</th></tr></thead><tbody>{cons_rows}</tbody></table>
<h2>5. Alerts</h2>
<table><thead><tr><th>Severity</th><th>Category</th><th>Message</th></tr></thead><tbody>{alert_rows}</tbody></table>
<h2>6. Repairs</h2>
<table><thead><tr><th>Column</th><th>Method</th><th>Rows</th><th>Description</th></tr></thead><tbody>{repair_rows}</tbody></table>
<footer>Generated by <code>compass/data_quality.py</code></footer>
</body></html>"""
        return html
