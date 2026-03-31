"""
Liquidity risk monitoring system.

Bid-ask spread tracking, market depth estimation, liquidity score (0-1),
liquidity regime classification (normal/stressed/crisis), capacity-adjusted
position sizing, illiquidity premium (Amihud ratio), stressed scenarios.

Generates an HTML report at reports/liquidity_risk.html.

Usage::

    from compass.liquidity_risk import LiquidityRiskMonitor
    monitor = LiquidityRiskMonitor(market_data)
    results = monitor.analyze()
    monitor.generate_report()
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

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "liquidity_risk.html"

LIQ_REGIMES = ("normal", "stressed", "crisis")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class LiquiditySnapshot:
    """Liquidity metrics at a single point in time."""
    date: str
    bid_ask_spread: float
    volume: float
    depth: float               # estimated market depth
    amihud: float              # Amihud illiquidity ratio
    score: float               # 0-1 composite
    regime: str


@dataclass
class LiquidityScore:
    """Composite liquidity score breakdown."""
    score: float               # 0 (illiquid) to 1 (liquid)
    spread_component: float
    volume_component: float
    impact_component: float
    regime: str


@dataclass
class LiquidityRegime:
    """Classified liquidity regime."""
    regime: str                # "normal", "stressed", "crisis"
    probability: float
    duration_days: int
    avg_score: float
    avg_spread: float


@dataclass
class CapacityResult:
    """Capacity-adjusted position sizing."""
    base_contracts: int
    adjusted_contracts: int
    capacity_factor: float     # 0-1, multiplier on base
    estimated_impact: float
    reason: str


@dataclass
class AmihudResult:
    """Amihud illiquidity ratio analysis."""
    current: float
    mean: float
    percentile: float
    z_score: float
    illiquidity_premium_bps: float


@dataclass
class StressScenario:
    """Stressed liquidity scenario."""
    name: str
    spread_multiplier: float
    volume_fraction: float
    impact_multiplier: float
    adjusted_score: float
    capacity_pct: float


@dataclass
class SpreadAlert:
    """Alert for abnormal spread."""
    date: str
    spread: float
    z_score: float
    severity: str              # "warning", "critical"


# ── Monitor ─────────────────────────────────────────────────────────────


class LiquidityRiskMonitor:
    """Liquidity risk monitoring and analysis."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        spread_col: str = "bid_ask_spread",
        volume_col: str = "volume",
        price_col: str = "close",
        returns_col: Optional[str] = None,
        base_position: int = 5,
        spread_alert_z: float = 2.0,
    ) -> None:
        self.data = market_data.copy()
        self.spread_col = spread_col
        self.volume_col = volume_col
        self.price_col = price_col
        self.base_position = base_position
        self.spread_alert_z = spread_alert_z

        # Ensure columns exist with defaults
        if spread_col not in self.data.columns:
            self.data[spread_col] = 0.05
        if volume_col not in self.data.columns:
            self.data[volume_col] = 10000.0
        if price_col not in self.data.columns:
            self.data[price_col] = 100.0

        # Compute returns if not provided
        if returns_col and returns_col in self.data.columns:
            self.returns = self.data[returns_col]
        else:
            self.returns = self.data[price_col].pct_change().fillna(0)

        # Results
        self.snapshots: List[LiquiditySnapshot] = []
        self.current_score: Optional[LiquidityScore] = None
        self.regime_history: List[LiquidityRegime] = []
        self.capacity: Optional[CapacityResult] = None
        self.amihud: Optional[AmihudResult] = None
        self.stress_scenarios: List[StressScenario] = []
        self.alerts: List[SpreadAlert] = []

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "LiquidityRiskMonitor":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.snapshots = self._compute_snapshots()
        self.current_score = self._current_liquidity_score()
        self.regime_history = self._classify_regimes()
        self.amihud = self._amihud_analysis()
        self.capacity = self._capacity_sizing()
        self.stress_scenarios = self._stress_scenarios()
        self.alerts = self._spread_alerts()
        return {
            "snapshots": self.snapshots,
            "current_score": self.current_score,
            "regime_history": self.regime_history,
            "amihud": self.amihud,
            "capacity": self.capacity,
            "stress_scenarios": self.stress_scenarios,
            "alerts": self.alerts,
        }

    # ── Snapshots ───────────────────────────────────────────────────────

    def _compute_snapshots(self) -> List[LiquiditySnapshot]:
        spreads = self.data[self.spread_col].values
        volumes = self.data[self.volume_col].values
        prices = self.data[self.price_col].values
        rets = self.returns.values

        results: List[LiquiditySnapshot] = []
        for i in range(len(self.data)):
            spread = float(spreads[i])
            vol = float(volumes[i])
            price = float(prices[i])
            ret = float(rets[i])

            depth = vol * price * 0.01  # rough depth estimate
            amihud = abs(ret) / (vol * price / 1e6) if vol * price > 0 else 0.0
            score = self._liquidity_score_single(spread, vol, amihud, spreads[:i+1], volumes[:i+1])
            regime = self._classify_single(score)

            date_val = str(self.data.index[i]) if hasattr(self.data.index, '__getitem__') else str(i)
            results.append(LiquiditySnapshot(
                date=date_val, bid_ask_spread=spread, volume=vol,
                depth=depth, amihud=amihud, score=score, regime=regime,
            ))
        return results

    def _liquidity_score_single(
        self, spread: float, volume: float, amihud: float,
        hist_spreads: np.ndarray, hist_volumes: np.ndarray,
    ) -> float:
        """Compute single-point liquidity score 0-1."""
        # Spread component: lower spread = better
        mean_sp = float(np.mean(hist_spreads)) if len(hist_spreads) > 0 else spread
        spread_comp = max(0, 1.0 - spread / (2 * mean_sp + 1e-10))

        # Volume component: higher volume = better
        mean_vol = float(np.mean(hist_volumes)) if len(hist_volumes) > 0 else volume
        volume_comp = min(volume / (mean_vol + 1e-10), 2.0) / 2.0

        # Impact component: lower Amihud = better
        impact_comp = max(0, 1.0 - min(amihud * 100, 1.0))

        return float(np.clip(0.40 * spread_comp + 0.35 * volume_comp + 0.25 * impact_comp, 0, 1))

    @staticmethod
    def _classify_single(score: float) -> str:
        if score >= 0.6:
            return "normal"
        elif score >= 0.3:
            return "stressed"
        else:
            return "crisis"

    # ── Current score ───────────────────────────────────────────────────

    def _current_liquidity_score(self) -> Optional[LiquidityScore]:
        if not self.snapshots:
            return None
        last = self.snapshots[-1]
        spreads = self.data[self.spread_col].values
        volumes = self.data[self.volume_col].values
        mean_sp = float(np.mean(spreads))
        mean_vol = float(np.mean(volumes))

        spread_comp = max(0, 1.0 - last.bid_ask_spread / (2 * mean_sp + 1e-10))
        vol_comp = min(last.volume / (mean_vol + 1e-10), 2.0) / 2.0
        impact_comp = max(0, 1.0 - min(last.amihud * 100, 1.0))

        return LiquidityScore(
            score=last.score, spread_component=spread_comp,
            volume_component=vol_comp, impact_component=impact_comp,
            regime=last.regime,
        )

    # ── Regime classification ───────────────────────────────────────────

    def _classify_regimes(self) -> List[LiquidityRegime]:
        if not self.snapshots:
            return []
        regimes_seen: Dict[str, List[LiquiditySnapshot]] = {}
        for s in self.snapshots:
            regimes_seen.setdefault(s.regime, []).append(s)

        results: List[LiquidityRegime] = []
        n = len(self.snapshots)
        for regime, snaps in regimes_seen.items():
            results.append(LiquidityRegime(
                regime=regime, probability=len(snaps) / n,
                duration_days=len(snaps),
                avg_score=float(np.mean([s.score for s in snaps])),
                avg_spread=float(np.mean([s.bid_ask_spread for s in snaps])),
            ))
        return sorted(results, key=lambda r: -r.probability)

    # ── Amihud analysis ─────────────────────────────────────────────────

    def _amihud_analysis(self) -> Optional[AmihudResult]:
        if not self.snapshots:
            return None
        amihuds = np.array([s.amihud for s in self.snapshots])
        amihuds = amihuds[~np.isnan(amihuds)]
        if len(amihuds) < 5:
            return None
        current = float(amihuds[-1])
        mean = float(np.mean(amihuds))
        std = float(np.std(amihuds))
        z = (current - mean) / std if std > 1e-10 else 0.0
        pct = float((amihuds < current).mean())
        # Illiquidity premium: rough bps estimate
        premium = float(np.clip(z * 5, 0, 50))

        return AmihudResult(
            current=current, mean=mean, percentile=pct,
            z_score=z, illiquidity_premium_bps=premium,
        )

    # ── Capacity sizing ─────────────────────────────────────────────────

    def _capacity_sizing(self) -> Optional[CapacityResult]:
        if not self.snapshots:
            return None
        last = self.snapshots[-1]
        score = last.score
        regime = last.regime

        if regime == "crisis":
            factor = 0.25
            reason = "Crisis regime: reduce to 25% of base"
        elif regime == "stressed":
            factor = 0.50
            reason = "Stressed regime: reduce to 50% of base"
        else:
            factor = min(score / 0.6, 1.0)
            reason = f"Normal regime: capacity factor {factor:.0%}"

        adjusted = max(1, int(self.base_position * factor))
        impact = last.amihud * adjusted * 100

        return CapacityResult(
            base_contracts=self.base_position,
            adjusted_contracts=adjusted,
            capacity_factor=factor,
            estimated_impact=impact,
            reason=reason,
        )

    # ── Stress scenarios ────────────────────────────────────────────────

    def _stress_scenarios(self) -> List[StressScenario]:
        if not self.snapshots:
            return []
        last = self.snapshots[-1]
        scenarios = [
            ("Normal", 1.0, 1.0, 1.0),
            ("Moderate Stress", 2.0, 0.5, 1.5),
            ("Severe Stress", 5.0, 0.2, 3.0),
            ("Flash Crash", 10.0, 0.05, 10.0),
            ("Market Close", 3.0, 0.1, 5.0),
        ]
        results: List[StressScenario] = []
        for name, sp_mult, vol_frac, imp_mult in scenarios:
            adj_spread = last.bid_ask_spread * sp_mult
            adj_vol = last.volume * vol_frac
            adj_amihud = last.amihud * imp_mult

            spreads = self.data[self.spread_col].values
            volumes = self.data[self.volume_col].values
            adj_score = self._liquidity_score_single(adj_spread, adj_vol, adj_amihud, spreads, volumes)
            cap_pct = max(adj_score / 0.6, 0.1) if adj_score < 0.6 else 1.0

            results.append(StressScenario(
                name=name, spread_multiplier=sp_mult,
                volume_fraction=vol_frac, impact_multiplier=imp_mult,
                adjusted_score=adj_score, capacity_pct=cap_pct,
            ))
        return results

    # ── Spread alerts ───────────────────────────────────────────────────

    def _spread_alerts(self) -> List[SpreadAlert]:
        spreads = self.data[self.spread_col].values
        if len(spreads) < 20:
            return []
        mean = float(np.mean(spreads))
        std = float(np.std(spreads))
        if std < 1e-10:
            return []

        alerts: List[SpreadAlert] = []
        for i in range(len(spreads)):
            z = (spreads[i] - mean) / std
            if z > self.spread_alert_z:
                severity = "critical" if z > self.spread_alert_z * 2 else "warning"
                date = str(self.data.index[i]) if hasattr(self.data.index, '__getitem__') else str(i)
                alerts.append(SpreadAlert(
                    date=date, spread=float(spreads[i]),
                    z_score=float(z), severity=severity,
                ))
        return alerts

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.current_score is None:
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
        charts: Dict[str, str] = {}
        charts["timeline"] = self._chart_timeline()
        charts["regime"] = self._chart_regime()
        charts["score"] = self._chart_score()
        return charts

    def _chart_timeline(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.snapshots:
            return ""
        xs = range(len(self.snapshots))
        spreads = [s.bid_ask_spread for s in self.snapshots]
        scores = [s.score for s in self.snapshots]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(xs, spreads, color="#f59e0b", lw=0.8); ax1.set_ylabel("Spread"); ax1.set_title("Bid-Ask Spread", fontsize=10)
        ax1.grid(True, alpha=0.2)
        regime_colors = {"normal": "#16a34a", "stressed": "#f59e0b", "crisis": "#dc2626"}
        for i in range(len(self.snapshots) - 1):
            c = regime_colors.get(self.snapshots[i].regime, "#64748b")
            ax2.axvspan(i, i+1, alpha=0.15, color=c)
        ax2.plot(xs, scores, color="#3b82f6", lw=0.8); ax2.set_ylabel("Score"); ax2.set_xlabel("Time")
        ax2.set_title("Liquidity Score", fontsize=10); ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_regime(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.regime_history:
            return ""
        names = [r.regime for r in self.regime_history]
        probs = [r.probability for r in self.regime_history]
        colors = {"normal": "#16a34a", "stressed": "#f59e0b", "crisis": "#dc2626"}
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(probs, labels=names, colors=[colors.get(n, "#64748b") for n in names],
               autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title("Regime Distribution", fontsize=11); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_score(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self.current_score is None:
            return ""
        cs = self.current_score
        fig, ax = plt.subplots(figsize=(5, 3))
        components = ["Spread", "Volume", "Impact", "Total"]
        vals = [cs.spread_component, cs.volume_component, cs.impact_component, cs.score]
        colors = ["#3b82f6", "#16a34a", "#f59e0b", "#1e293b"]
        ax.barh(components, vals, color=colors, alpha=0.85)
        ax.set_xlim(0, 1); ax.set_xlabel("Score (0-1)")
        ax.set_title("Liquidity Score Breakdown", fontsize=11); ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        cs = self.current_score or LiquidityScore(0, 0, 0, 0, "crisis")
        cap = self.capacity or CapacityResult(0, 0, 0, 0, "")
        am = self.amihud or AmihudResult(0, 0, 0, 0, 0)

        score_cls = "good" if cs.score >= 0.6 else "bad" if cs.score < 0.3 else ""
        regime_colors = {"normal": "#16a34a", "stressed": "#f59e0b", "crisis": "#dc2626"}
        rc = regime_colors.get(cs.regime, "#64748b")

        regime_rows = ""
        for r in self.regime_history:
            regime_rows += f'<tr><td>{r.regime}</td><td>{r.probability:.0%}</td><td>{r.duration_days}d</td><td>{r.avg_score:.3f}</td><td>{r.avg_spread:.4f}</td></tr>\n'

        stress_rows = ""
        for s in self.stress_scenarios:
            cls = "bad" if s.adjusted_score < 0.3 else ""
            stress_rows += f'<tr><td>{s.name}</td><td>{s.spread_multiplier:.0f}x</td><td>{s.volume_fraction:.0%}</td><td class="{cls}">{s.adjusted_score:.3f}</td><td>{s.capacity_pct:.0%}</td></tr>\n'

        alert_rows = ""
        for a in self.alerts[-20:]:
            cls = "bad" if a.severity == "critical" else ""
            alert_rows += f'<tr><td>{a.date}</td><td>{a.spread:.4f}</td><td>{a.z_score:.1f}</td><td class="{cls}">{a.severity}</td></tr>\n'
        if not alert_rows:
            alert_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No alerts</td></tr>'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Liquidity Risk Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }} .warn {{ color:#f59e0b; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  .risk-badge {{ display:inline-block; padding:0.3em 0.8em; border-radius:4px; color:white; font-weight:700; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Liquidity Risk Dashboard</h1>
<div class="meta">{len(self.snapshots)} observations &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {score_cls}">{cs.score:.2f}</div><div class="label">Liquidity Score</div></div>
  <div class="kpi"><div class="value"><span class="risk-badge" style="background:{rc}">{cs.regime.upper()}</span></div><div class="label">Regime</div></div>
  <div class="kpi"><div class="value">{cap.adjusted_contracts}</div><div class="label">Capacity (contracts)</div></div>
  <div class="kpi"><div class="value">{am.illiquidity_premium_bps:.0f}bp</div><div class="label">Illiquidity Premium</div></div>
  <div class="kpi"><div class="value">{len(self.alerts)}</div><div class="label">Spread Alerts</div></div>
</div>
<h2>1. Liquidity Timeline</h2>{_img("timeline")}
<h2>2. Score Breakdown</h2>{_img("score")}
<h2>3. Regime Distribution</h2>{_img("regime")}
<table><thead><tr><th>Regime</th><th>Probability</th><th>Duration</th><th>Avg Score</th><th>Avg Spread</th></tr></thead><tbody>{regime_rows}</tbody></table>
<h2>4. Amihud Illiquidity</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Current</td><td>{am.current:.6f}</td></tr>
<tr><td>Mean</td><td>{am.mean:.6f}</td></tr>
<tr><td>Percentile</td><td>{am.percentile:.0%}</td></tr>
<tr><td>Z-Score</td><td>{am.z_score:+.2f}</td></tr>
<tr><td>Premium (bps)</td><td>{am.illiquidity_premium_bps:.1f}</td></tr>
</tbody></table>
<h2>5. Capacity Sizing</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Base Position</td><td>{cap.base_contracts}</td></tr>
<tr><td>Adjusted</td><td>{cap.adjusted_contracts}</td></tr>
<tr><td>Capacity Factor</td><td>{cap.capacity_factor:.0%}</td></tr>
<tr><td>Reason</td><td style="text-align:left">{cap.reason}</td></tr>
</tbody></table>
<h2>6. Stress Scenarios</h2>
<table><thead><tr><th>Scenario</th><th>Spread Mult</th><th>Volume</th><th>Adj Score</th><th>Capacity</th></tr></thead><tbody>{stress_rows}</tbody></table>
<h2>7. Spread Alerts</h2>
<table><thead><tr><th>Date</th><th>Spread</th><th>Z-Score</th><th>Severity</th></tr></thead><tbody>{alert_rows}</tbody></table>
<footer>Generated by <code>compass/liquidity_risk.py</code></footer>
</body></html>"""
        return html
