"""Drawdown recovery prediction engine – survival analysis, regime-conditional
recovery expectations, drawdown episode clustering, and position sizing
adjustment during drawdowns.

Provides:
  1. Historical drawdown analysis (depth, duration, recovery time)
  2. Recovery probability estimation via Kaplan-Meier survival analysis
  3. Regime-conditional recovery expectations
  4. Drawdown clustering (find similar historical episodes)
  5. Conditional expected recovery time given current depth
  6. Position sizing adjustment during drawdowns
  7. HTML report with waterfall, survival curves, heatmap, clusters
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
class DrawdownEpisode:
    """A single historical drawdown episode."""
    episode_id: int
    start_idx: int
    trough_idx: int
    end_idx: int
    depth: float            # positive fraction (0.15 = 15%)
    decline_days: int
    recovery_days: int
    total_days: int
    recovered: bool
    regime: str = ""
    post_recovery_return: float = 0.0  # return in 20 bars after recovery


@dataclass
class SurvivalPoint:
    """One point on the Kaplan-Meier survival curve."""
    time: int               # days since trough
    survival_prob: float    # P(still in drawdown at this time)
    n_at_risk: int
    n_events: int           # recovered at this time


@dataclass
class RegimeRecovery:
    """Recovery statistics for one regime."""
    regime: str
    n_episodes: int
    avg_depth: float
    avg_recovery_days: float
    median_recovery_days: float
    recovery_rate: float       # fraction that recovered
    avg_post_recovery_return: float


@dataclass
class EpisodeCluster:
    """A cluster of similar drawdown episodes."""
    cluster_id: int
    n_episodes: int
    avg_depth: float
    avg_recovery_days: float
    depth_range: Tuple[float, float]
    regime_distribution: Dict[str, int]
    episode_ids: List[int]


@dataclass
class SizingAdjustment:
    """Position sizing recommendation during a drawdown."""
    current_depth: float
    recommended_scale: float   # 0-1 multiplier
    reasoning: str
    expected_recovery_days: int
    recovery_probability: float


@dataclass
class RecoveryResult:
    """Complete recovery analysis output."""
    episodes: List[DrawdownEpisode] = field(default_factory=list)
    survival_curve: List[SurvivalPoint] = field(default_factory=list)
    regime_recoveries: List[RegimeRecovery] = field(default_factory=list)
    clusters: List[EpisodeCluster] = field(default_factory=list)
    sizing_adjustment: Optional[SizingAdjustment] = None
    conditional_recovery: Dict[str, float] = field(default_factory=dict)
    n_episodes: int = 0
    overall_recovery_rate: float = 0.0
    generated_at: str = ""


# ── Core engine ─────────────────────────────────────────────────────────────
class DrawdownRecoveryPredictor:
    """Predicts drawdown recovery using survival analysis and clustering."""

    def __init__(
        self,
        min_dd_depth: float = 0.01,
        n_clusters: int = 3,
        sizing_floor: float = 0.25,
        sizing_dd_start: float = 0.05,
        sizing_dd_full: float = 0.25,
    ) -> None:
        self.min_dd_depth = min_dd_depth
        self.n_clusters = n_clusters
        self.sizing_floor = sizing_floor
        self.sizing_dd_start = sizing_dd_start
        self.sizing_dd_full = sizing_dd_full

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        returns: pd.Series,
        regimes: Optional[pd.Series] = None,
        current_depth: Optional[float] = None,
    ) -> RecoveryResult:
        """Run full recovery analysis.

        Parameters
        ----------
        returns : pd.Series
            Daily strategy returns.
        regimes : pd.Series, optional
            Regime labels aligned to same index.
        current_depth : float, optional
            Current drawdown depth (positive fraction) for sizing recommendation.
        """
        returns = returns.dropna()
        if len(returns) < 20:
            return RecoveryResult(generated_at=self._now())

        equity = (1 + returns).cumprod()
        episodes = self._extract_episodes(equity, returns, regimes)

        if not episodes:
            return RecoveryResult(generated_at=self._now())

        survival = self._kaplan_meier(episodes)
        regime_rec = self._regime_recovery(episodes)
        clusters = self._cluster_episodes(episodes)
        cond_recovery = self._conditional_recovery(episodes)

        sizing: Optional[SizingAdjustment] = None
        if current_depth is not None:
            sizing = self._sizing_recommendation(current_depth, episodes, survival)

        recovered = [e for e in episodes if e.recovered]
        rate = len(recovered) / len(episodes) if episodes else 0.0

        return RecoveryResult(
            episodes=episodes,
            survival_curve=survival,
            regime_recoveries=regime_rec,
            clusters=clusters,
            sizing_adjustment=sizing,
            conditional_recovery=cond_recovery,
            n_episodes=len(episodes),
            overall_recovery_rate=rate,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: RecoveryResult,
        output_path: str | Path = "reports/drawdown_recovery.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Recovery report written to %s", path)
        return path

    # ── Episode extraction ──────────────────────────────────────────────────
    def _extract_episodes(
        self,
        equity: pd.Series,
        returns: pd.Series,
        regimes: Optional[pd.Series],
    ) -> List[DrawdownEpisode]:
        running_max = equity.cummax()
        uw = (equity - running_max) / running_max
        uw_vals = uw.values
        n = len(uw_vals)
        episodes: List[DrawdownEpisode] = []
        eid = 0
        i = 0

        while i < n:
            if uw_vals[i] >= 0:
                i += 1
                continue

            start = max(0, i - 1)
            j = i
            trough = i
            while j < n and uw_vals[j] < 0:
                if uw_vals[j] < uw_vals[trough]:
                    trough = j
                j += 1

            end = j - 1 if j < n else n - 1
            recovered = j < n
            depth = abs(float(uw_vals[trough]))

            if depth < self.min_dd_depth:
                i = j
                continue

            regime = ""
            if regimes is not None and trough < len(regimes):
                regime = str(regimes.iloc[trough])

            # Post-recovery return (20 bars after recovery)
            post_ret = 0.0
            if recovered and end + 20 < n:
                post_ret = float(returns.iloc[end + 1:end + 21].sum())

            episodes.append(DrawdownEpisode(
                episode_id=eid,
                start_idx=start,
                trough_idx=trough,
                end_idx=end,
                depth=depth,
                decline_days=trough - start,
                recovery_days=max(0, end - trough),
                total_days=end - start,
                recovered=recovered,
                regime=regime,
                post_recovery_return=post_ret,
            ))
            eid += 1
            i = j

        return episodes

    # ── Kaplan-Meier survival ───────────────────────────────────────────────
    @staticmethod
    def _kaplan_meier(episodes: List[DrawdownEpisode]) -> List[SurvivalPoint]:
        """Estimate P(still in drawdown) at each time point."""
        # Recovery times; censored if not recovered
        times: List[Tuple[int, bool]] = []
        for e in episodes:
            if e.recovered:
                times.append((e.recovery_days, True))
            else:
                times.append((e.total_days, False))

        if not times:
            return []

        times.sort(key=lambda t: t[0])
        n_total = len(times)

        unique_times = sorted(set(t for t, _ in times))
        curve: List[SurvivalPoint] = []
        survival = 1.0
        at_risk = n_total

        # Time 0
        curve.append(SurvivalPoint(time=0, survival_prob=1.0, n_at_risk=n_total, n_events=0))

        for t in unique_times:
            events = sum(1 for tt, ev in times if tt == t and ev)
            censored = sum(1 for tt, ev in times if tt == t and not ev)

            if at_risk > 0 and events > 0:
                survival *= (1 - events / at_risk)

            curve.append(SurvivalPoint(
                time=t,
                survival_prob=max(0.0, survival),
                n_at_risk=at_risk,
                n_events=events,
            ))
            at_risk -= (events + censored)
            if at_risk <= 0:
                break

        return curve

    # ── Regime-conditional recovery ─────────────────────────────────────────
    @staticmethod
    def _regime_recovery(episodes: List[DrawdownEpisode]) -> List[RegimeRecovery]:
        by_regime: Dict[str, List[DrawdownEpisode]] = {}
        for e in episodes:
            r = e.regime or "unknown"
            by_regime.setdefault(r, []).append(e)

        results: List[RegimeRecovery] = []
        for regime, eps in sorted(by_regime.items()):
            recovered = [e for e in eps if e.recovered and e.recovery_days > 0]
            rec_days = [e.recovery_days for e in recovered]
            depths = [e.depth for e in eps]
            post_rets = [e.post_recovery_return for e in recovered if e.post_recovery_return != 0]

            results.append(RegimeRecovery(
                regime=regime,
                n_episodes=len(eps),
                avg_depth=float(np.mean(depths)),
                avg_recovery_days=float(np.mean(rec_days)) if rec_days else 0.0,
                median_recovery_days=float(np.median(rec_days)) if rec_days else 0.0,
                recovery_rate=len(recovered) / len(eps) if eps else 0.0,
                avg_post_recovery_return=float(np.mean(post_rets)) if post_rets else 0.0,
            ))
        return results

    # ── Clustering ──────────────────────────────────────────────────────────
    def _cluster_episodes(self, episodes: List[DrawdownEpisode]) -> List[EpisodeCluster]:
        if len(episodes) < self.n_clusters:
            return []

        depths = np.array([e.depth for e in episodes])

        # Simple quantile-based clustering on depth
        quantiles = np.linspace(0, 100, self.n_clusters + 1)
        boundaries = np.percentile(depths, quantiles)

        clusters: List[EpisodeCluster] = []
        for c in range(self.n_clusters):
            lo = boundaries[c]
            hi = boundaries[c + 1]
            if c == self.n_clusters - 1:
                mask = (depths >= lo) & (depths <= hi)
            else:
                mask = (depths >= lo) & (depths < hi)

            member_ids = [episodes[i].episode_id for i in range(len(episodes)) if mask[i]]
            members = [episodes[i] for i in range(len(episodes)) if mask[i]]

            if not members:
                continue

            rec_days = [e.recovery_days for e in members if e.recovered and e.recovery_days > 0]
            regime_dist: Dict[str, int] = {}
            for e in members:
                r = e.regime or "unknown"
                regime_dist[r] = regime_dist.get(r, 0) + 1

            clusters.append(EpisodeCluster(
                cluster_id=c,
                n_episodes=len(members),
                avg_depth=float(np.mean([e.depth for e in members])),
                avg_recovery_days=float(np.mean(rec_days)) if rec_days else 0.0,
                depth_range=(float(lo), float(hi)),
                regime_distribution=regime_dist,
                episode_ids=member_ids,
            ))
        return clusters

    # ── Conditional recovery time ───────────────────────────────────────────
    @staticmethod
    def _conditional_recovery(episodes: List[DrawdownEpisode]) -> Dict[str, float]:
        """Expected recovery days by depth bucket."""
        buckets = {"0-5%": (0, 0.05), "5-10%": (0.05, 0.10), "10-20%": (0.10, 0.20), "20%+": (0.20, 1.0)}
        result: Dict[str, float] = {}
        for label, (lo, hi) in buckets.items():
            matching = [
                e.recovery_days for e in episodes
                if e.recovered and lo <= e.depth < hi and e.recovery_days > 0
            ]
            result[label] = float(np.mean(matching)) if matching else 0.0
        return result

    # ── Sizing adjustment ───────────────────────────────────────────────────
    def _sizing_recommendation(
        self,
        current_depth: float,
        episodes: List[DrawdownEpisode],
        survival: List[SurvivalPoint],
    ) -> SizingAdjustment:
        # Scale: linear from 1.0 at sizing_dd_start to sizing_floor at sizing_dd_full
        if current_depth < self.sizing_dd_start:
            scale = 1.0
            reason = "Drawdown below threshold — full position sizing"
        elif current_depth >= self.sizing_dd_full:
            scale = self.sizing_floor
            reason = f"Deep drawdown ({current_depth:.0%}) — reduce to {self.sizing_floor:.0%}"
        else:
            frac = (current_depth - self.sizing_dd_start) / (self.sizing_dd_full - self.sizing_dd_start)
            scale = 1.0 - frac * (1.0 - self.sizing_floor)
            reason = f"Moderate drawdown ({current_depth:.0%}) — scale to {scale:.0%}"

        # Expected recovery from similar episodes
        similar = [e for e in episodes if e.recovered and abs(e.depth - current_depth) < 0.05]
        if similar:
            exp_days = int(np.mean([e.recovery_days for e in similar]))
        else:
            exp_days = int(np.mean([e.recovery_days for e in episodes if e.recovered])) if any(e.recovered for e in episodes) else 0

        # Recovery probability from survival curve
        rec_prob = 1.0
        if survival:
            last = survival[-1]
            rec_prob = 1.0 - last.survival_prob

        return SizingAdjustment(
            current_depth=current_depth,
            recommended_scale=round(scale, 2),
            reasoning=reason,
            expected_recovery_days=exp_days,
            recovery_probability=rec_prob,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: RecoveryResult) -> str:
        cards = self._html_cards(r)
        waterfall = self._svg_waterfall(r.episodes)
        survival_svg = self._svg_survival(r.survival_curve)
        heatmap = self._svg_recovery_heatmap(r.conditional_recovery)
        regime_tbl = self._html_regime_table(r.regime_recoveries)
        cluster_tbl = self._html_clusters(r.clusters)
        sizing_sec = self._html_sizing(r.sizing_adjustment)
        episodes_tbl = self._html_episodes(r.episodes)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Drawdown Recovery Analysis</title>
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
<h1>Drawdown Recovery Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_episodes} episodes</p>

{cards}

<div class="sec"><h2>Drawdown Waterfall</h2>{waterfall}</div>
<div class="sec"><h2>Survival Curve (Kaplan-Meier)</h2>{survival_svg}</div>
<div class="sec"><h2>Recovery Heatmap by Depth</h2>{heatmap}</div>

{regime_tbl}
{cluster_tbl}
{sizing_sec}
{episodes_tbl}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: RecoveryResult) -> str:
        avg_depth = float(np.mean([e.depth for e in r.episodes])) if r.episodes else 0
        rec = [e for e in r.episodes if e.recovered]
        avg_rec = float(np.mean([e.recovery_days for e in rec])) if rec else 0
        sizing = r.sizing_adjustment
        scale = f"{sizing.recommended_scale:.0%}" if sizing else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Episodes</div><div class="val">{r.n_episodes}</div></div>
<div class="card"><div class="lbl">Recovery Rate</div><div class="val">{r.overall_recovery_rate:.0%}</div></div>
<div class="card"><div class="lbl">Avg Depth</div><div class="val neg">{avg_depth:.1%}</div></div>
<div class="card"><div class="lbl">Avg Recovery</div><div class="val">{avg_rec:.0f}d</div></div>
<div class="card"><div class="lbl">Position Scale</div><div class="val">{scale}</div></div>
</div>"""

    @staticmethod
    def _svg_waterfall(episodes: List[DrawdownEpisode]) -> str:
        if not episodes:
            return "<p>No episodes.</p>"
        top = sorted(episodes, key=lambda e: -e.depth)[:15]
        w, h = 520, 30 * len(top) + 30
        pl = 60
        max_d = max(e.depth for e in top) or 0.01
        bars = ""
        for i, e in enumerate(top):
            y = 10 + i * 30
            bw = e.depth / max_d * (w - pl - 60)
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="10" fill="#94a3b8">#{e.episode_id}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="20" rx="3" fill="#f87171" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#e2e8f0">{e.depth:.1%} ({e.recovery_days}d rec)</text>'
            )
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _svg_survival(curve: List[SurvivalPoint]) -> str:
        if not curve:
            return "<p>No data.</p>"
        w, h = 450, 220
        pl, pb, pt = 50, 30, 15
        cw, ch = w - pl, h - pb - pt
        max_t = max(c.time for c in curve) or 1

        pts = []
        for c in curve:
            x = pl + c.time / max_t * cw
            y = pt + ch - c.survival_prob * ch
            pts.append(f"{x:.0f},{y:.0f}")

        polyline = " ".join(pts)
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{pt + ch}" x2="{w}" y2="{pt + ch}" stroke="#475569" stroke-width="1"/>'
            f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt + ch}" stroke="#475569" stroke-width="1"/>'
            f'<polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="2"/>'
            f'<text x="{pl - 5}" y="{pt + 4}" text-anchor="end" font-size="10" fill="#94a3b8">100%</text>'
            f'<text x="{pl - 5}" y="{pt + ch}" text-anchor="end" font-size="10" fill="#94a3b8">0%</text>'
            f'<text x="{w // 2}" y="{h - 3}" text-anchor="middle" font-size="10" fill="#94a3b8">Days since trough</text>'
            f'</svg>'
        )

    @staticmethod
    def _svg_recovery_heatmap(cond: Dict[str, float]) -> str:
        if not cond:
            return "<p>No data.</p>"
        w, h = 400, 80
        n = len(cond)
        cell_w = (w - 20) / n
        cells = ""
        max_d = max(cond.values()) or 1
        for i, (label, days) in enumerate(cond.items()):
            x = 10 + i * cell_w
            intensity = min(255, int(days / max_d * 200))
            colour = f"rgb({intensity + 50},{30},{30})"
            cells += (
                f'<rect x="{x:.0f}" y="5" width="{cell_w:.0f}" height="40" fill="{colour}" '
                f'stroke="#0f172a" stroke-width="1" rx="3"/>'
                f'<text x="{x + cell_w / 2:.0f}" y="30" text-anchor="middle" font-size="11" fill="#e2e8f0">'
                f'{days:.0f}d</text>'
                f'<text x="{x + cell_w / 2:.0f}" y="65" text-anchor="middle" font-size="10" fill="#94a3b8">'
                f'{label}</text>'
            )
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{cells}</svg>'

    @staticmethod
    def _html_regime_table(regimes: List[RegimeRecovery]) -> str:
        if not regimes:
            return ""
        rows = ""
        for rr in regimes:
            rows += (
                f"<tr><td>{rr.regime}</td><td>{rr.n_episodes}</td>"
                f"<td>{rr.avg_depth:.1%}</td>"
                f"<td>{rr.avg_recovery_days:.0f}</td>"
                f"<td>{rr.median_recovery_days:.0f}</td>"
                f"<td>{rr.recovery_rate:.0%}</td>"
                f"<td>{rr.avg_post_recovery_return:.2%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Regime-Conditional Recovery</h2>
<table>
<thead><tr><th>Regime</th><th>Episodes</th><th>Avg Depth</th><th>Avg Rec</th><th>Med Rec</th><th>Rec Rate</th><th>Post-Rec Return</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_clusters(clusters: List[EpisodeCluster]) -> str:
        if not clusters:
            return ""
        rows = ""
        for c in clusters:
            regimes = ", ".join(f"{k}:{v}" for k, v in sorted(c.regime_distribution.items()))
            rows += (
                f"<tr><td>Cluster {c.cluster_id}</td><td>{c.n_episodes}</td>"
                f"<td>{c.avg_depth:.1%}</td>"
                f"<td>{c.depth_range[0]:.1%}–{c.depth_range[1]:.1%}</td>"
                f"<td>{c.avg_recovery_days:.0f}d</td>"
                f"<td>{regimes}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Episode Clusters</h2>
<table>
<thead><tr><th>Cluster</th><th>Episodes</th><th>Avg Depth</th><th>Depth Range</th><th>Avg Rec</th><th>Regimes</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_sizing(sizing: Optional[SizingAdjustment]) -> str:
        if not sizing:
            return ""
        return f"""<div class="sec">
<h2>Position Sizing Adjustment</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Current Depth</td><td class="neg">{sizing.current_depth:.1%}</td></tr>
<tr><td>Recommended Scale</td><td class="warn">{sizing.recommended_scale:.0%}</td></tr>
<tr><td>Expected Recovery</td><td>{sizing.expected_recovery_days}d</td></tr>
<tr><td>Recovery Probability</td><td>{sizing.recovery_probability:.0%}</td></tr>
<tr><td>Reasoning</td><td>{sizing.reasoning}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_episodes(episodes: List[DrawdownEpisode]) -> str:
        if not episodes:
            return ""
        top = sorted(episodes, key=lambda e: -e.depth)[:20]
        rows = ""
        for e in top:
            rows += (
                f"<tr><td>{e.episode_id}</td>"
                f"<td class='neg'>{e.depth:.1%}</td>"
                f"<td>{e.decline_days}d</td>"
                f"<td>{e.recovery_days}d</td>"
                f"<td>{e.total_days}d</td>"
                f"<td>{'Yes' if e.recovered else 'No'}</td>"
                f"<td>{e.regime or 'N/A'}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Top Episodes</h2>
<table>
<thead><tr><th>#</th><th>Depth</th><th>Decline</th><th>Recovery</th><th>Total</th><th>Recovered</th><th>Regime</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
