"""
Backtest reconciliation tool — compares backtest vs paper trading results.

Identifies trade-level discrepancies, categorises root causes (slippage,
fill quality, data staleness, timing drift, model divergence), and computes
a 0-100 reconciliation score.  Produces a self-contained HTML report at
reports/reconciliation.html.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.backtest_reconciler import BacktestReconciler
    rec = BacktestReconciler(backtest_trades, paper_trades)
    result = rec.reconcile()
    BacktestReconciler.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "reconciliation.html"

# ── Root-cause categories ────────────────────────────────────────────────

ROOT_CAUSES = (
    "slippage",
    "fill_quality",
    "data_staleness",
    "timing_drift",
    "model_divergence",
)

# ── Thresholds (configurable) ────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "entry_price_tol_pct": 0.5,   # 0.5 % tolerance for entry price match
    "exit_price_tol_pct": 0.5,
    "pnl_tol_dollars": 20.0,      # $20 tolerance for P&L match
    "timing_tol_minutes": 30.0,   # 30 min tolerance for entry/exit time
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TradeDiscrepancy:
    """A single discrepancy between backtest and paper trade."""

    trade_id: str
    field_name: str           # e.g. "entry_price", "exit_price", "pnl", "timing"
    backtest_value: float
    paper_value: float
    diff: float
    diff_pct: float
    root_cause: str
    severity: str             # "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "field_name": self.field_name,
            "backtest_value": self.backtest_value,
            "paper_value": self.paper_value,
            "diff": self.diff,
            "diff_pct": self.diff_pct,
            "root_cause": self.root_cause,
            "severity": self.severity,
        }


@dataclass
class TradePairComparison:
    """Side-by-side comparison of one matched trade pair."""

    trade_id: str
    bt_entry_price: float
    pp_entry_price: float
    bt_exit_price: float
    pp_exit_price: float
    bt_pnl: float
    pp_pnl: float
    bt_entry_time: Optional[pd.Timestamp]
    pp_entry_time: Optional[pd.Timestamp]
    bt_exit_time: Optional[pd.Timestamp]
    pp_exit_time: Optional[pd.Timestamp]
    entry_price_diff_pct: float
    exit_price_diff_pct: float
    pnl_diff: float
    timing_diff_minutes: float
    discrepancies: List[TradeDiscrepancy] = field(default_factory=list)
    is_matched: bool = True


@dataclass
class RootCauseSummary:
    """Aggregate root-cause breakdown."""

    cause: str
    count: int
    pct_of_total: float
    avg_severity_score: float  # 1=low, 2=medium, 3=high


@dataclass
class ReconciliationResult:
    """Full result from reconciliation."""

    n_backtest_trades: int
    n_paper_trades: int
    n_matched: int
    n_unmatched_bt: int
    n_unmatched_pp: int
    comparisons: List[TradePairComparison]
    discrepancies: List[TradeDiscrepancy]
    root_cause_summary: List[RootCauseSummary]
    reconciliation_score: float       # 0-100
    score_breakdown: Dict[str, float]
    thresholds: Dict[str, float]


# ── Root-cause classification ────────────────────────────────────────────


def classify_root_cause(
    field_name: str,
    diff_pct: float,
    timing_diff_min: float,
) -> str:
    """Classify the most likely root cause of a discrepancy."""
    abs_diff = abs(diff_pct)

    if field_name in ("entry_price", "exit_price"):
        if abs_diff < 1.0:
            return "slippage"
        elif abs_diff < 3.0:
            return "fill_quality"
        else:
            return "data_staleness"

    if field_name == "timing":
        if abs(timing_diff_min) < 60:
            return "timing_drift"
        else:
            return "model_divergence"

    if field_name == "pnl":
        if abs(timing_diff_min) > 30:
            return "timing_drift"
        elif abs_diff < 5.0:
            return "slippage"
        elif abs_diff < 20.0:
            return "fill_quality"
        else:
            return "model_divergence"

    return "model_divergence"


def classify_severity(diff_pct: float, field_name: str) -> str:
    """Classify severity of a discrepancy."""
    abs_d = abs(diff_pct)
    if field_name == "pnl":
        if abs_d < 5.0:
            return "low"
        elif abs_d < 20.0:
            return "medium"
        return "high"
    if field_name == "timing":
        if abs_d < 30:
            return "low"
        elif abs_d < 120:
            return "medium"
        return "high"
    # price fields
    if abs_d < 0.5:
        return "low"
    elif abs_d < 2.0:
        return "medium"
    return "high"


SEVERITY_SCORE = {"low": 1, "medium": 2, "high": 3}


# ── Matching engine ──────────────────────────────────────────────────────


def match_trades(
    bt: pd.DataFrame,
    pp: pd.DataFrame,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match backtest trades to paper trades by trade_id or nearest date.

    Returns:
        (matched_pairs, unmatched_bt_indices, unmatched_pp_indices)
    """
    if "trade_id" in bt.columns and "trade_id" in pp.columns:
        bt_ids = dict(zip(bt["trade_id"], bt.index))
        pp_ids = dict(zip(pp["trade_id"], pp.index))
        common = set(bt_ids.keys()) & set(pp_ids.keys())
        pairs = [(bt_ids[k], pp_ids[k]) for k in common]
        unmatched_bt = [i for k, i in bt_ids.items() if k not in common]
        unmatched_pp = [i for k, i in pp_ids.items() if k not in common]
        return pairs, unmatched_bt, unmatched_pp

    # Fallback: match by nearest entry date
    bt_dates = pd.to_datetime(bt["entry_date"])
    pp_dates = pd.to_datetime(pp["entry_date"])
    used_pp: set = set()
    pairs: List[Tuple[int, int]] = []
    unmatched_bt: List[int] = []

    for bi in bt.index:
        best_j: Optional[int] = None
        best_diff = pd.Timedelta.max
        for pj in pp.index:
            if pj in used_pp:
                continue
            diff = abs(bt_dates[bi] - pp_dates[pj])
            if diff < best_diff:
                best_diff = diff
                best_j = pj
        if best_j is not None and best_diff <= pd.Timedelta(days=2):
            pairs.append((bi, best_j))
            used_pp.add(best_j)
        else:
            unmatched_bt.append(bi)

    unmatched_pp = [j for j in pp.index if j not in used_pp]
    return pairs, unmatched_bt, unmatched_pp


# ── Score computation ────────────────────────────────────────────────────


def compute_reconciliation_score(
    comparisons: List[TradePairComparison],
    n_bt: int,
    n_pp: int,
    thresholds: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Compute 0-100 reconciliation score.

    Components (each 0-25):
      - match_rate:  % of trades successfully matched
      - price_accuracy:  how close entry/exit prices are
      - pnl_accuracy:  how close P&L values are
      - timing_accuracy:  how close entry/exit timing is
    """
    if n_bt == 0 and n_pp == 0:
        return 100.0, {"match_rate": 25.0, "price_accuracy": 25.0,
                        "pnl_accuracy": 25.0, "timing_accuracy": 25.0}

    total_possible = max(n_bt, n_pp, 1)
    n_matched = len(comparisons)

    # 1) Match rate (0-25)
    match_rate = min(25.0, 25.0 * n_matched / total_possible)

    if not comparisons:
        return match_rate, {"match_rate": match_rate, "price_accuracy": 0.0,
                            "pnl_accuracy": 0.0, "timing_accuracy": 0.0}

    # 2) Price accuracy (0-25)
    entry_diffs = [abs(c.entry_price_diff_pct) for c in comparisons]
    exit_diffs = [abs(c.exit_price_diff_pct) for c in comparisons]
    price_tol = thresholds["entry_price_tol_pct"]
    within_entry = sum(1 for d in entry_diffs if d <= price_tol)
    within_exit = sum(1 for d in exit_diffs if d <= price_tol)
    price_accuracy = 25.0 * (within_entry + within_exit) / (2 * n_matched)

    # 3) P&L accuracy (0-25)
    pnl_tol = thresholds["pnl_tol_dollars"]
    within_pnl = sum(1 for c in comparisons if abs(c.pnl_diff) <= pnl_tol)
    pnl_accuracy = 25.0 * within_pnl / n_matched

    # 4) Timing accuracy (0-25)
    time_tol = thresholds["timing_tol_minutes"]
    within_time = sum(
        1 for c in comparisons if abs(c.timing_diff_minutes) <= time_tol
    )
    timing_accuracy = 25.0 * within_time / n_matched

    total = match_rate + price_accuracy + pnl_accuracy + timing_accuracy
    breakdown = {
        "match_rate": round(match_rate, 2),
        "price_accuracy": round(price_accuracy, 2),
        "pnl_accuracy": round(pnl_accuracy, 2),
        "timing_accuracy": round(timing_accuracy, 2),
    }
    return round(total, 2), breakdown


# ── Core reconciler ──────────────────────────────────────────────────────


class BacktestReconciler:
    """Compare backtest results vs paper trading trade-by-trade."""

    def __init__(
        self,
        backtest_trades: pd.DataFrame,
        paper_trades: pd.DataFrame,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            backtest_trades: DataFrame with columns: entry_price, exit_price,
                pnl, entry_date, exit_date.  Optional: trade_id.
            paper_trades: Same schema as backtest_trades.
            thresholds: Override default tolerance thresholds.
        """
        required = {"entry_price", "exit_price", "pnl", "entry_date", "exit_date"}
        for name, df in [("backtest", backtest_trades), ("paper", paper_trades)]:
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{name} trades missing columns: {missing}")

        self.bt = backtest_trades.copy()
        self.pp = paper_trades.copy()
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def _compare_pair(
        self, bt_row: pd.Series, pp_row: pd.Series, trade_id: str
    ) -> TradePairComparison:
        """Compare a single matched trade pair."""
        bt_entry = float(bt_row["entry_price"])
        pp_entry = float(pp_row["entry_price"])
        bt_exit = float(bt_row["exit_price"])
        pp_exit = float(pp_row["exit_price"])
        bt_pnl = float(bt_row["pnl"])
        pp_pnl = float(pp_row["pnl"])

        bt_entry_t = pd.Timestamp(bt_row["entry_date"])
        pp_entry_t = pd.Timestamp(pp_row["entry_date"])
        bt_exit_t = pd.Timestamp(bt_row["exit_date"])
        pp_exit_t = pd.Timestamp(pp_row["exit_date"])

        def _pct_diff(a: float, b: float) -> float:
            base = abs(a) if abs(a) > 1e-9 else 1.0
            return (b - a) / base * 100.0

        entry_diff_pct = _pct_diff(bt_entry, pp_entry)
        exit_diff_pct = _pct_diff(bt_exit, pp_exit)
        pnl_diff = pp_pnl - bt_pnl

        timing_diff_min = 0.0
        try:
            td = abs(pp_entry_t - bt_entry_t)
            timing_diff_min = td.total_seconds() / 60.0
        except Exception:
            pass

        discrepancies: List[TradeDiscrepancy] = []

        # Check entry price
        if abs(entry_diff_pct) > self.thresholds["entry_price_tol_pct"]:
            sev = classify_severity(entry_diff_pct, "entry_price")
            discrepancies.append(TradeDiscrepancy(
                trade_id=trade_id, field_name="entry_price",
                backtest_value=bt_entry, paper_value=pp_entry,
                diff=pp_entry - bt_entry, diff_pct=entry_diff_pct,
                root_cause=classify_root_cause("entry_price", entry_diff_pct, timing_diff_min),
                severity=sev,
            ))

        # Check exit price
        if abs(exit_diff_pct) > self.thresholds["exit_price_tol_pct"]:
            sev = classify_severity(exit_diff_pct, "exit_price")
            discrepancies.append(TradeDiscrepancy(
                trade_id=trade_id, field_name="exit_price",
                backtest_value=bt_exit, paper_value=pp_exit,
                diff=pp_exit - bt_exit, diff_pct=exit_diff_pct,
                root_cause=classify_root_cause("exit_price", exit_diff_pct, timing_diff_min),
                severity=sev,
            ))

        # Check P&L
        pnl_diff_pct = _pct_diff(bt_pnl, pp_pnl) if abs(bt_pnl) > 1e-9 else 0.0
        if abs(pnl_diff) > self.thresholds["pnl_tol_dollars"]:
            sev = classify_severity(pnl_diff_pct, "pnl")
            discrepancies.append(TradeDiscrepancy(
                trade_id=trade_id, field_name="pnl",
                backtest_value=bt_pnl, paper_value=pp_pnl,
                diff=pnl_diff, diff_pct=pnl_diff_pct,
                root_cause=classify_root_cause("pnl", pnl_diff_pct, timing_diff_min),
                severity=sev,
            ))

        # Check timing
        if timing_diff_min > self.thresholds["timing_tol_minutes"]:
            sev = classify_severity(timing_diff_min, "timing")
            discrepancies.append(TradeDiscrepancy(
                trade_id=trade_id, field_name="timing",
                backtest_value=0.0, paper_value=timing_diff_min,
                diff=timing_diff_min, diff_pct=timing_diff_min,
                root_cause=classify_root_cause("timing", 0, timing_diff_min),
                severity=sev,
            ))

        return TradePairComparison(
            trade_id=trade_id,
            bt_entry_price=bt_entry, pp_entry_price=pp_entry,
            bt_exit_price=bt_exit, pp_exit_price=pp_exit,
            bt_pnl=bt_pnl, pp_pnl=pp_pnl,
            bt_entry_time=bt_entry_t, pp_entry_time=pp_entry_t,
            bt_exit_time=bt_exit_t, pp_exit_time=pp_exit_t,
            entry_price_diff_pct=entry_diff_pct,
            exit_price_diff_pct=exit_diff_pct,
            pnl_diff=pnl_diff,
            timing_diff_minutes=timing_diff_min,
            discrepancies=discrepancies,
        )

    def _build_root_cause_summary(
        self, discrepancies: List[TradeDiscrepancy]
    ) -> List[RootCauseSummary]:
        if not discrepancies:
            return []

        counts: Counter = Counter(d.root_cause for d in discrepancies)
        severity_sums: Dict[str, List[int]] = {}
        for d in discrepancies:
            severity_sums.setdefault(d.root_cause, []).append(
                SEVERITY_SCORE[d.severity]
            )

        total = len(discrepancies)
        summaries = []
        for cause in ROOT_CAUSES:
            if cause not in counts:
                continue
            scores = severity_sums[cause]
            summaries.append(RootCauseSummary(
                cause=cause,
                count=counts[cause],
                pct_of_total=counts[cause] / total,
                avg_severity_score=float(np.mean(scores)),
            ))
        return sorted(summaries, key=lambda s: s.count, reverse=True)

    def reconcile(self) -> ReconciliationResult:
        """Run full reconciliation."""
        pairs, unmatched_bt, unmatched_pp = match_trades(self.bt, self.pp)

        comparisons: List[TradePairComparison] = []
        all_discrepancies: List[TradeDiscrepancy] = []

        for bi, pi in pairs:
            bt_row = self.bt.iloc[bi] if isinstance(bi, int) else self.bt.loc[bi]
            pp_row = self.pp.iloc[pi] if isinstance(pi, int) else self.pp.loc[pi]
            tid = str(bt_row.get("trade_id", f"pair-{bi}-{pi}"))
            comp = self._compare_pair(bt_row, pp_row, tid)
            comparisons.append(comp)
            all_discrepancies.extend(comp.discrepancies)

        root_summary = self._build_root_cause_summary(all_discrepancies)
        score, breakdown = compute_reconciliation_score(
            comparisons, len(self.bt), len(self.pp), self.thresholds
        )

        return ReconciliationResult(
            n_backtest_trades=len(self.bt),
            n_paper_trades=len(self.pp),
            n_matched=len(pairs),
            n_unmatched_bt=len(unmatched_bt),
            n_unmatched_pp=len(unmatched_pp),
            comparisons=comparisons,
            discrepancies=all_discrepancies,
            root_cause_summary=root_summary,
            reconciliation_score=score,
            score_breakdown=breakdown,
            thresholds=self.thresholds,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: ReconciliationResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML helpers ─────────────────────────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _score_color(score: float) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 60:
        return "#d29922"
    return "#f85149"


def _score_card(result: ReconciliationResult) -> str:
    color = _score_color(result.reconciliation_score)
    bd = result.score_breakdown

    def _bar(label: str, val: float) -> str:
        pct = val / 25.0 * 100
        return (
            f'<div class="score-row"><span class="label">{label}</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<span class="value">{val:.1f}/25</span></div>'
        )

    return f"""
    <div class="score-card">
      <div class="big-score" style="color:{color}">{result.reconciliation_score:.0f}</div>
      <div class="score-label">Reconciliation Score</div>
      {_bar("Match Rate", bd["match_rate"])}
      {_bar("Price Accuracy", bd["price_accuracy"])}
      {_bar("P&L Accuracy", bd["pnl_accuracy"])}
      {_bar("Timing Accuracy", bd["timing_accuracy"])}
    </div>"""


def _comparison_table(comparisons: List[TradePairComparison]) -> str:
    if not comparisons:
        return "<p>No matched trades to compare.</p>"
    rows = ""
    for c in comparisons[:100]:  # cap for report size
        n_disc = len(c.discrepancies)
        cls = "row-ok" if n_disc == 0 else ("row-warn" if n_disc <= 2 else "row-bad")
        rows += f"""<tr class="{cls}">
          <td>{c.trade_id}</td>
          <td>{c.bt_entry_price:.2f}</td><td>{c.pp_entry_price:.2f}</td>
          <td>{_fmt_pct(c.entry_price_diff_pct)}</td>
          <td>{c.bt_pnl:.2f}</td><td>{c.pp_pnl:.2f}</td>
          <td>{_fmt_dollar(c.pnl_diff)}</td>
          <td>{c.timing_diff_minutes:.0f}m</td>
          <td>{n_disc}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr><th>Trade</th><th>BT Entry</th><th>PP Entry</th><th>Entry Δ%</th>
          <th>BT P&L</th><th>PP P&L</th><th>P&L Δ</th>
          <th>Time Δ</th><th>Issues</th></tr>
      {rows}
    </table>"""


def _histogram_svg(values: List[float], title: str, xlabel: str) -> str:
    if not values:
        return f"<p>No data for {title}.</p>"
    w, h = 600, 280
    pad = 60
    arr = np.array(values)
    n_bins = min(30, max(5, len(arr) // 5))
    counts, edges = np.histogram(arr, bins=n_bins)
    max_c = max(counts) if len(counts) else 1
    cw = (w - 2 * pad) / len(counts)
    ch = h - 2 * pad
    bars = []
    for i, c in enumerate(counts):
        bh = (c / max_c) * ch if max_c else 0
        x = pad + i * cw
        y = h - pad - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw - 1:.1f}" '
                     f'height="{bh:.1f}" fill="#58a6ff" opacity="0.8"/>')
    return f"""
    <svg viewBox="0 0 {w} {h}" class="chart">
      <text x="{w//2}" y="20" text-anchor="middle" class="svg-title">{title}</text>
      <text x="{w//2}" y="{h-5}" text-anchor="middle" class="svg-label">{xlabel}</text>
      {"".join(bars)}
    </svg>"""


def _pie_svg(summaries: List[RootCauseSummary]) -> str:
    if not summaries:
        return "<p>No discrepancies found — perfect reconciliation.</p>"
    w, h = 400, 400
    cx, cy, r = w // 2, h // 2 - 20, 140
    colors = ["#58a6ff", "#f0883e", "#f85149", "#3fb950", "#bc8cff"]
    slices = []
    start = 0.0
    legend_items = []
    for i, s in enumerate(summaries):
        angle = s.pct_of_total * 2 * math.pi
        end = start + angle
        large = 1 if angle > math.pi else 0
        x1 = cx + r * math.cos(start)
        y1 = cy + r * math.sin(start)
        x2 = cx + r * math.cos(end)
        y2 = cy + r * math.sin(end)
        color = colors[i % len(colors)]
        slices.append(
            f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} '
            f'A{r},{r} 0 {large} 1 {x2:.1f},{y2:.1f} Z" fill="{color}" opacity="0.85"/>'
        )
        legend_items.append(
            f'<text x="{w - 130}" y="{h - 60 + i * 18}" class="svg-label" fill="{color}">'
            f'● {s.cause} ({s.count})</text>'
        )
        start = end
    return f"""
    <svg viewBox="0 0 {w} {h}" class="chart">
      <text x="{w//2}" y="20" text-anchor="middle" class="svg-title">Root Cause Breakdown</text>
      {"".join(slices)}
      {"".join(legend_items)}
    </svg>"""


def _build_html(result: ReconciliationResult) -> str:
    pnl_diffs = [abs(c.pnl_diff) for c in result.comparisons]
    entry_diffs = [abs(c.entry_price_diff_pct) for c in result.comparisons]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Backtest Reconciliation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .top-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .score-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                 padding: 24px; text-align: center; }}
  .big-score {{ font-size: 4em; font-weight: 800; }}
  .score-label {{ color: #8b949e; font-size: 1.1em; margin-bottom: 16px; }}
  .score-row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .score-row .label {{ width: 120px; text-align: right; color: #8b949e; font-size: 0.85em; }}
  .score-row .value {{ width: 60px; font-weight: 600; font-size: 0.85em; }}
  .bar-bg {{ flex: 1; height: 8px; background: #21262d; border-radius: 4px; }}
  .bar-fill {{ height: 100%; background: #58a6ff; border-radius: 4px; }}
  .stats {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
            padding: 24px; }}
  .stats div {{ margin: 8px 0; }}
  .stats .label {{ color: #8b949e; }}
  .stats .value {{ font-weight: 600; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 16px 0;
                       font-size: 0.9em; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; text-align: right; }}
  table.data-table td:first-child, table.data-table th:first-child {{ text-align: left; }}
  .row-ok td {{ color: #c9d1d9; }}
  .row-warn td {{ color: #d29922; }}
  .row-bad td {{ color: #f85149; }}
  .chart {{ width: 100%; max-width: 650px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 14px; }}
  .svg-label {{ fill: #8b949e; font-size: 11px; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Backtest Reconciliation Report</h1>
<p class="meta">{result.n_matched} matched trades out of {result.n_backtest_trades} backtest /
   {result.n_paper_trades} paper &middot; {len(result.discrepancies)} discrepancies found</p>

<div class="top-grid">
  {_score_card(result)}
  <div class="stats">
    <h3>Summary</h3>
    <div><span class="label">Backtest trades: </span><span class="value">{result.n_backtest_trades}</span></div>
    <div><span class="label">Paper trades: </span><span class="value">{result.n_paper_trades}</span></div>
    <div><span class="label">Matched: </span><span class="value">{result.n_matched}</span></div>
    <div><span class="label">Unmatched (BT): </span><span class="value">{result.n_unmatched_bt}</span></div>
    <div><span class="label">Unmatched (PP): </span><span class="value">{result.n_unmatched_pp}</span></div>
    <div><span class="label">Discrepancies: </span><span class="value">{len(result.discrepancies)}</span></div>
  </div>
</div>

<h2>Trade-Level Comparison</h2>
{_comparison_table(result.comparisons)}

<h2>Discrepancy Analysis</h2>
<div class="charts-grid">
  {_histogram_svg(pnl_diffs, "P&L Discrepancy Distribution", "|P&L diff| ($)")}
  {_pie_svg(result.root_cause_summary)}
</div>

{_histogram_svg(entry_diffs, "Entry Price Discrepancy Distribution", "|Entry price diff| (%)")}

</body>
</html>"""
