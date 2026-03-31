"""Drawdown analyzer – comprehensive drawdown analysis with regime attribution,
conditional drawdown-at-risk, clustering detection, recovery speed analysis,
and risk-adjusted return ratios.

Provides:
  1. Max DD, avg DD, DD duration, recovery time, underwater curve
  2. Drawdown decomposition by regime
  3. Conditional Drawdown-at-Risk (CDaR) at 95th and 99th percentile
  4. Drawdown clustering (temporal autocorrelation of DD events)
  5. Recovery speed analysis with predictive indicators
  6. Calmar, Sterling, and Burke ratio computation
  7. HTML report with charts and tables
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class DrawdownEvent:
    """A single drawdown episode from peak to recovery."""
    start_idx: int
    trough_idx: int
    end_idx: int              # index where equity recovers (or last index)
    depth: float              # max DD as positive fraction (e.g. 0.15 = 15%)
    duration_days: int        # peak-to-recovery in bars
    decline_days: int         # peak-to-trough
    recovery_days: int        # trough-to-recovery
    recovered: bool           # True if equity fully recovered
    regime: str = ""          # regime label at trough


@dataclass
class RegimeAttribution:
    """Drawdown attribution for one regime."""
    regime: str
    n_events: int
    avg_depth: float
    max_depth: float
    avg_duration: int
    total_time_underwater_pct: float  # fraction of regime bars in DD
    contribution_pct: float           # share of total DD-days


@dataclass
class RecoveryStats:
    """Recovery speed analysis."""
    n_recoveries: int
    avg_recovery_days: int
    median_recovery_days: int
    fast_recovery_threshold: int      # 25th percentile
    slow_recovery_threshold: int      # 75th percentile
    avg_depth_fast: float             # avg DD of fast recoveries
    avg_depth_slow: float             # avg DD of slow recoveries


@dataclass
class ClusteringResult:
    """Drawdown clustering analysis."""
    autocorrelation_lag1: float       # DD series lag-1 autocorrelation
    clustering_score: float           # 0-100 (100 = heavily clustered)
    avg_gap_between_dds: float        # avg bars between DD starts
    interpretation: str


@dataclass
class RiskRatios:
    """Risk-adjusted return ratios."""
    calmar: float       # CAGR / max DD
    sterling: float     # CAGR / avg(top 5 DDs)
    burke: float        # CAGR / sqrt(sum(DD²))
    cagr: float
    max_dd: float


@dataclass
class DrawdownResult:
    """Complete drawdown analysis output."""
    max_dd: float = 0.0
    avg_dd: float = 0.0
    cdar_95: float = 0.0               # Conditional DD-at-Risk 95%
    cdar_99: float = 0.0
    events: List[DrawdownEvent] = field(default_factory=list)
    underwater_curve: Optional[pd.Series] = None
    regime_attribution: List[RegimeAttribution] = field(default_factory=list)
    recovery: Optional[RecoveryStats] = None
    clustering: Optional[ClusteringResult] = None
    ratios: Optional[RiskRatios] = None
    n_bars: int = 0
    generated_at: str = ""


# ── Core analyzer ───────────────────────────────────────────────────────────
class DrawdownAnalyzer:
    """Comprehensive drawdown analysis engine."""

    def __init__(
        self,
        cdar_levels: Tuple[float, ...] = (0.95, 0.99),
        min_dd_depth: float = 0.001,
    ) -> None:
        self.cdar_levels = cdar_levels
        self.min_dd_depth = min_dd_depth

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        returns: pd.Series,
        regimes: Optional[pd.Series] = None,
    ) -> DrawdownResult:
        """Run full drawdown analysis.

        Parameters
        ----------
        returns : pd.Series
            Daily (or per-bar) strategy returns.
        regimes : pd.Series, optional
            Regime labels aligned to same index as returns.
        """
        returns = returns.dropna()
        if len(returns) < 5:
            return DrawdownResult(generated_at=self._now())

        equity = (1 + returns).cumprod()
        underwater = self._underwater_curve(equity)
        events = self._find_events(underwater, regimes)

        max_dd = float(underwater.min())
        avg_dd = float(underwater[underwater < 0].mean()) if (underwater < 0).any() else 0.0

        cdar_95, cdar_99 = self._compute_cdar(underwater)

        regime_attr: List[RegimeAttribution] = []
        if regimes is not None:
            regimes = regimes.reindex(returns.index).fillna("unknown")
            regime_attr = self._regime_attribution(underwater, events, regimes)

        recovery = self._recovery_analysis(events)
        clustering = self._clustering_analysis(underwater, events)
        ratios = self._compute_ratios(returns, events, max_dd)

        return DrawdownResult(
            max_dd=max_dd,
            avg_dd=avg_dd,
            cdar_95=cdar_95,
            cdar_99=cdar_99,
            events=events,
            underwater_curve=underwater,
            regime_attribution=regime_attr,
            recovery=recovery,
            clustering=clustering,
            ratios=ratios,
            n_bars=len(returns),
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: DrawdownResult,
        output_path: str | Path = "reports/drawdown_analysis.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Drawdown report written to %s", path)
        return path

    # ── Underwater curve ────────────────────────────────────────────────────
    @staticmethod
    def _underwater_curve(equity: pd.Series) -> pd.Series:
        running_max = equity.cummax()
        return (equity - running_max) / running_max

    # ── Event detection ─────────────────────────────────────────────────────
    def _find_events(
        self,
        underwater: pd.Series,
        regimes: Optional[pd.Series],
    ) -> List[DrawdownEvent]:
        events: List[DrawdownEvent] = []
        uw = underwater.values
        n = len(uw)
        i = 0

        while i < n:
            if uw[i] >= 0:
                i += 1
                continue
            # Start of DD
            start = max(0, i - 1)
            # Find trough
            j = i
            trough = i
            while j < n and uw[j] < 0:
                if uw[j] < uw[trough]:
                    trough = j
                j += 1
            end = j - 1 if j < n else n - 1
            recovered = j < n

            depth = abs(float(uw[trough]))
            if depth < self.min_dd_depth:
                i = j
                continue

            regime = ""
            if regimes is not None and trough < len(regimes):
                regime = str(regimes.iloc[trough])

            events.append(DrawdownEvent(
                start_idx=start,
                trough_idx=trough,
                end_idx=end,
                depth=depth,
                duration_days=end - start,
                decline_days=trough - start,
                recovery_days=max(0, end - trough),
                recovered=recovered,
                regime=regime,
            ))
            i = j

        return events

    # ── CDaR ────────────────────────────────────────────────────────────────
    def _compute_cdar(self, underwater: pd.Series) -> Tuple[float, float]:
        dd_values = -underwater.values  # positive = deeper DD
        dd_values = dd_values[dd_values > 0]
        if len(dd_values) < 5:
            return (0.0, 0.0)

        results = []
        for level in self.cdar_levels:
            var = float(np.percentile(dd_values, level * 100))
            tail = dd_values[dd_values >= var]
            cdar = float(np.mean(tail)) if len(tail) > 0 else var
            results.append(cdar)

        return (results[0] if len(results) > 0 else 0.0,
                results[1] if len(results) > 1 else 0.0)

    # ── Regime attribution ──────────────────────────────────────────────────
    @staticmethod
    def _regime_attribution(
        underwater: pd.Series,
        events: List[DrawdownEvent],
        regimes: pd.Series,
    ) -> List[RegimeAttribution]:
        # Group events by regime
        regime_events: Dict[str, List[DrawdownEvent]] = {}
        for e in events:
            r = e.regime or "unknown"
            regime_events.setdefault(r, []).append(e)

        total_dd_days = sum(e.duration_days for e in events) or 1

        # Per-regime time underwater
        uw_arr = underwater.values < 0
        reg_arr = regimes.values

        results: List[RegimeAttribution] = []
        for regime in sorted(regime_events):
            evts = regime_events[regime]
            depths = [e.depth for e in evts]
            durations = [e.duration_days for e in evts]

            # Time underwater in this regime
            mask = reg_arr == regime
            regime_bars = int(np.sum(mask))
            underwater_bars = int(np.sum(uw_arr & mask))
            uw_pct = underwater_bars / regime_bars if regime_bars > 0 else 0.0

            contrib = sum(durations) / total_dd_days

            results.append(RegimeAttribution(
                regime=regime,
                n_events=len(evts),
                avg_depth=float(np.mean(depths)),
                max_depth=float(np.max(depths)),
                avg_duration=int(np.mean(durations)),
                total_time_underwater_pct=uw_pct,
                contribution_pct=contrib,
            ))

        return results

    # ── Recovery analysis ───────────────────────────────────────────────────
    @staticmethod
    def _recovery_analysis(events: List[DrawdownEvent]) -> RecoveryStats:
        recovered = [e for e in events if e.recovered and e.recovery_days > 0]
        if not recovered:
            return RecoveryStats(0, 0, 0, 0, 0, 0.0, 0.0)

        rec_days = np.array([e.recovery_days for e in recovered])
        depths = np.array([e.depth for e in recovered])

        p25 = int(np.percentile(rec_days, 25))
        p75 = int(np.percentile(rec_days, 75))

        fast_mask = rec_days <= p25
        slow_mask = rec_days >= p75

        return RecoveryStats(
            n_recoveries=len(recovered),
            avg_recovery_days=int(np.mean(rec_days)),
            median_recovery_days=int(np.median(rec_days)),
            fast_recovery_threshold=p25,
            slow_recovery_threshold=p75,
            avg_depth_fast=float(np.mean(depths[fast_mask])) if fast_mask.any() else 0.0,
            avg_depth_slow=float(np.mean(depths[slow_mask])) if slow_mask.any() else 0.0,
        )

    # ── Clustering ──────────────────────────────────────────────────────────
    @staticmethod
    def _clustering_analysis(
        underwater: pd.Series, events: List[DrawdownEvent],
    ) -> ClusteringResult:
        # DD indicator series (1 if underwater, 0 if not)
        dd_ind = (underwater.values < 0).astype(float)
        n = len(dd_ind)

        # Lag-1 autocorrelation of DD indicator
        if n < 10:
            return ClusteringResult(0.0, 0.0, 0.0, "Insufficient data")

        mean_d = np.mean(dd_ind)
        var_d = np.var(dd_ind)
        if var_d < 1e-12:
            return ClusteringResult(0.0, 0.0, 0.0, "No drawdowns detected")

        centered = dd_ind - mean_d
        ac1 = float(np.sum(centered[1:] * centered[:-1]) / ((n - 1) * var_d))

        # Clustering score: 0-100 based on autocorrelation
        score = max(0.0, min(100.0, ac1 * 100.0))

        # Average gap between DD starts
        starts = [e.start_idx for e in events]
        if len(starts) > 1:
            gaps = np.diff(starts)
            avg_gap = float(np.mean(gaps))
        else:
            avg_gap = float(n)

        if score > 70:
            interp = "Heavy clustering — drawdowns strongly cluster in time"
        elif score > 40:
            interp = "Moderate clustering — some temporal dependence"
        else:
            interp = "Low clustering — drawdowns appear relatively independent"

        return ClusteringResult(
            autocorrelation_lag1=ac1,
            clustering_score=score,
            avg_gap_between_dds=avg_gap,
            interpretation=interp,
        )

    # ── Risk ratios ─────────────────────────────────────────────────────────
    @staticmethod
    def _compute_ratios(
        returns: pd.Series,
        events: List[DrawdownEvent],
        max_dd: float,
    ) -> RiskRatios:
        n = len(returns)
        total_ret = float((1 + returns).prod())
        years = n / PERIODS_PER_YEAR
        cagr = total_ret ** (1 / years) - 1 if years > 0 and total_ret > 0 else 0.0

        abs_max_dd = abs(max_dd)

        # Calmar = CAGR / max DD
        calmar = cagr / abs_max_dd if abs_max_dd > 1e-9 else 0.0

        # Sterling = CAGR / avg(top N DDs)
        depths = sorted([e.depth for e in events], reverse=True)
        top5 = depths[:5] if len(depths) >= 5 else depths
        avg_top = float(np.mean(top5)) if top5 else 0.0
        sterling = cagr / avg_top if avg_top > 1e-9 else 0.0

        # Burke = CAGR / sqrt(sum(DD²))
        if depths:
            sum_sq = float(np.sum(np.array(depths) ** 2))
            burke = cagr / math.sqrt(sum_sq) if sum_sq > 1e-12 else 0.0
        else:
            burke = 0.0

        return RiskRatios(
            calmar=calmar,
            sterling=sterling,
            burke=burke,
            cagr=cagr,
            max_dd=abs_max_dd,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: DrawdownResult) -> str:
        cards = self._html_cards(r)
        uw_chart = self._svg_underwater(r.underwater_curve)
        dd_dist = self._svg_dd_distribution(r.events)
        regime_tbl = self._html_regime_table(r.regime_attribution)
        recovery_sec = self._html_recovery(r.recovery)
        cluster_sec = self._html_clustering(r.clustering)
        ratios_tbl = self._html_ratios(r.ratios)
        events_tbl = self._html_events(r.events)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Drawdown Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}.warn{{color:#fbbf24}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Drawdown Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_bars} bars &middot; {len(r.events)} DD events</p>

{cards}

<div class="sec">
<h2>Underwater Curve</h2>
{uw_chart}
</div>

<div class="sec">
<h2>Drawdown Distribution</h2>
{dd_dist}
</div>

{regime_tbl}
{recovery_sec}
{cluster_sec}
{ratios_tbl}
{events_tbl}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: DrawdownResult) -> str:
        rat = r.ratios
        calmar = f"{rat.calmar:.2f}" if rat else "N/A"
        cagr = f"{rat.cagr:.1%}" if rat else "N/A"
        rec = r.recovery
        avg_rec = f"{rec.avg_recovery_days}d" if rec and rec.n_recoveries > 0 else "N/A"
        clust = f"{r.clustering.clustering_score:.0f}/100" if r.clustering else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Max Drawdown</div><div class="val neg">{r.max_dd:.2%}</div></div>
<div class="card"><div class="lbl">Avg Drawdown</div><div class="val">{r.avg_dd:.2%}</div></div>
<div class="card"><div class="lbl">CDaR 95%</div><div class="val">{r.cdar_95:.2%}</div></div>
<div class="card"><div class="lbl">CDaR 99%</div><div class="val">{r.cdar_99:.2%}</div></div>
<div class="card"><div class="lbl">CAGR</div><div class="val">{cagr}</div></div>
<div class="card"><div class="lbl">Calmar Ratio</div><div class="val">{calmar}</div></div>
<div class="card"><div class="lbl">Avg Recovery</div><div class="val">{avg_rec}</div></div>
<div class="card"><div class="lbl">Clustering</div><div class="val">{clust}</div></div>
</div>"""

    @staticmethod
    def _svg_underwater(uw: Optional[pd.Series]) -> str:
        if uw is None or uw.empty:
            return "<p>No data.</p>"
        w, h = 600, 180
        pl, pb, pt = 50, 30, 10
        cw, ch = w - pl, h - pb - pt
        vals = uw.values
        n = len(vals)
        min_v = min(vals.min(), -0.001)

        pts = []
        for i in range(n):
            x = pl + i / max(n - 1, 1) * cw
            y = pt + (1 - vals[i] / min_v) * ch if min_v != 0 else pt + ch
            pts.append(f"{x:.1f},{y:.1f}")

        baseline_y = pt
        poly = f"{pl},{baseline_y} " + " ".join(pts) + f" {pl + cw},{baseline_y}"

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<polygon points="{poly}" fill="#f87171" opacity="0.3"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#f87171" stroke-width="1.5"/>'
            f'<line x1="{pl}" y1="{baseline_y}" x2="{w}" y2="{baseline_y}" stroke="#475569" stroke-width="1"/>'
            f'<text x="{pl - 5}" y="{pt + ch}" text-anchor="end" font-size="10" fill="#94a3b8">{min_v:.1%}</text>'
            f'<text x="{pl - 5}" y="{pt + 4}" text-anchor="end" font-size="10" fill="#94a3b8">0%</text>'
            f'</svg>'
        )

    @staticmethod
    def _svg_dd_distribution(events: List[DrawdownEvent]) -> str:
        if not events:
            return "<p>No drawdown events.</p>"
        depths = [e.depth * 100 for e in events]
        w, h = 500, 160
        pl, pb, pt = 50, 35, 10
        cw, ch = w - pl, h - pb - pt

        # Simple histogram: 10 bins
        bins = np.linspace(0, max(depths) * 1.01, 11)
        counts, _ = np.histogram(depths, bins=bins)
        max_c = max(counts) or 1
        n_bins = len(counts)
        bw = cw / n_bins - 2

        bars = ""
        for i in range(n_bins):
            x = pl + i * (cw / n_bins) + 1
            bar_h = counts[i] / max_c * ch
            y = pt + ch - bar_h
            bars += (
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{bar_h:.0f}" '
                f'rx="2" fill="#f97316" opacity="0.7"/>'
            )
        # axis labels
        bars += f'<text x="{pl}" y="{h - 5}" font-size="10" fill="#94a3b8">0%</text>'
        bars += f'<text x="{w - 20}" y="{h - 5}" font-size="10" fill="#94a3b8">{max(depths):.0f}%</text>'

        baseline = pt + ch
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{baseline}" x2="{w}" y2="{baseline}" stroke="#475569" stroke-width="1"/>'
            f'{bars}</svg>'
        )

    @staticmethod
    def _html_regime_table(attr: List[RegimeAttribution]) -> str:
        if not attr:
            return ""
        rows = ""
        for a in sorted(attr, key=lambda x: -x.max_depth):
            rows += (
                f"<tr><td>{a.regime}</td><td>{a.n_events}</td>"
                f"<td>{a.avg_depth:.2%}</td><td class='neg'>{a.max_depth:.2%}</td>"
                f"<td>{a.avg_duration}d</td>"
                f"<td>{a.total_time_underwater_pct:.1%}</td>"
                f"<td>{a.contribution_pct:.1%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Regime Attribution</h2>
<table>
<thead><tr><th>Regime</th><th>Events</th><th>Avg Depth</th><th>Max Depth</th><th>Avg Duration</th><th>Time Underwater</th><th>Contribution</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_recovery(rec: Optional[RecoveryStats]) -> str:
        if not rec or rec.n_recoveries == 0:
            return ""
        return f"""<div class="sec">
<h2>Recovery Analysis</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Recoveries Observed</td><td>{rec.n_recoveries}</td></tr>
<tr><td>Avg Recovery Days</td><td>{rec.avg_recovery_days}</td></tr>
<tr><td>Median Recovery Days</td><td>{rec.median_recovery_days}</td></tr>
<tr><td>Fast Recovery Threshold (P25)</td><td>{rec.fast_recovery_threshold}d</td></tr>
<tr><td>Slow Recovery Threshold (P75)</td><td>{rec.slow_recovery_threshold}d</td></tr>
<tr><td>Avg Depth — Fast Recoveries</td><td>{rec.avg_depth_fast:.2%}</td></tr>
<tr><td>Avg Depth — Slow Recoveries</td><td>{rec.avg_depth_slow:.2%}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_clustering(cl: Optional[ClusteringResult]) -> str:
        if not cl:
            return ""
        cls = "neg" if cl.clustering_score > 60 else "warn" if cl.clustering_score > 30 else "pos"
        return f"""<div class="sec">
<h2>Drawdown Clustering</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Lag-1 Autocorrelation</td><td>{cl.autocorrelation_lag1:.4f}</td></tr>
<tr><td>Clustering Score</td><td class="{cls}">{cl.clustering_score:.0f}/100</td></tr>
<tr><td>Avg Gap Between DDs</td><td>{cl.avg_gap_between_dds:.0f} bars</td></tr>
<tr><td>Interpretation</td><td>{cl.interpretation}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_ratios(rat: Optional[RiskRatios]) -> str:
        if not rat:
            return ""
        return f"""<div class="sec">
<h2>Risk-Adjusted Ratios</h2>
<table>
<thead><tr><th>Ratio</th><th>Value</th><th>Description</th></tr></thead>
<tbody>
<tr><td>Calmar</td><td>{rat.calmar:.3f}</td><td>CAGR / Max DD</td></tr>
<tr><td>Sterling</td><td>{rat.sterling:.3f}</td><td>CAGR / Avg Top-5 DDs</td></tr>
<tr><td>Burke</td><td>{rat.burke:.3f}</td><td>CAGR / sqrt(sum DD²)</td></tr>
<tr><td>CAGR</td><td>{rat.cagr:.2%}</td><td></td></tr>
<tr><td>Max DD</td><td class="neg">{rat.max_dd:.2%}</td><td></td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_events(events: List[DrawdownEvent]) -> str:
        if not events:
            return ""
        top = sorted(events, key=lambda e: -e.depth)[:15]
        rows = ""
        for i, e in enumerate(top, 1):
            rows += (
                f"<tr><td>{i}</td>"
                f"<td class='neg'>{e.depth:.2%}</td>"
                f"<td>{e.decline_days}d</td>"
                f"<td>{e.recovery_days}d</td>"
                f"<td>{e.duration_days}d</td>"
                f"<td>{'Yes' if e.recovered else 'No'}</td>"
                f"<td>{e.regime or 'N/A'}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Top Drawdown Events</h2>
<table>
<thead><tr><th>#</th><th>Depth</th><th>Decline</th><th>Recovery</th><th>Duration</th><th>Recovered</th><th>Regime</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
