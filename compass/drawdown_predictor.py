"""
Drawdown recovery prediction engine.

Classifies drawdowns by severity and estimates recovery time using
historical analogues, regime conditioning, and Monte Carlo simulation.

Severity buckets:
  SHALLOW       < 5%
  MODERATE      5-10%
  DEEP          10-20%
  CATASTROPHIC  > 20%

All methods operate on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class DrawdownSeverity(str, Enum):
    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"
    CATASTROPHIC = "catastrophic"


SEVERITY_THRESHOLDS = {
    DrawdownSeverity.SHALLOW: (0.0, 0.05),
    DrawdownSeverity.MODERATE: (0.05, 0.10),
    DrawdownSeverity.DEEP: (0.10, 0.20),
    DrawdownSeverity.CATASTROPHIC: (0.20, float("inf")),
}

SEVERITY_SIZE_MULT = {
    DrawdownSeverity.SHALLOW: 1.0,
    DrawdownSeverity.MODERATE: 0.70,
    DrawdownSeverity.DEEP: 0.40,
    DrawdownSeverity.CATASTROPHIC: 0.10,
}


@dataclass
class DrawdownEpisode:
    """A single historical drawdown episode."""
    start_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime]
    peak_equity: float
    trough_equity: float
    max_drawdown: float
    severity: DrawdownSeverity
    recovery_days: Optional[int] = None
    regime_at_trough: Optional[str] = None


@dataclass
class RecoveryEstimate:
    """Recovery time prediction."""
    current_drawdown: float
    severity: DrawdownSeverity
    expected_days: float
    median_days: float
    pct_80_days: float
    pct_95_days: float
    n_analogues: int
    size_multiplier: float


@dataclass
class RegimeRecoveryProfile:
    """Recovery stats conditioned on market regime."""
    regime: str
    n_episodes: int
    avg_recovery_days: float
    median_recovery_days: float
    avg_drawdown: float


@dataclass
class MonteCarloRecovery:
    """Monte Carlo simulation of recovery paths."""
    current_drawdown: float
    n_simulations: int
    expected_days: float
    median_days: float
    pct_80_days: float
    pct_95_days: float
    prob_recover_30d: float
    prob_recover_60d: float
    prob_recover_90d: float
    confidence_bands: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class EarlyWarning:
    """Early warning signal."""
    signal_type: str
    value: float
    threshold: float
    triggered: bool
    message: str


@dataclass
class DrawdownPrediction:
    """Full prediction result."""
    current_drawdown: float
    severity: DrawdownSeverity
    recovery: RecoveryEstimate
    monte_carlo: Optional[MonteCarloRecovery] = None
    warnings: List[EarlyWarning] = field(default_factory=list)
    regime_profiles: List[RegimeRecoveryProfile] = field(default_factory=list)
    size_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class DrawdownPredictor:
    """Drawdown recovery prediction engine.

    Args:
        mc_simulations: Number of Monte Carlo paths.
        mc_horizon: Max days to simulate recovery.
        velocity_window: Days for drawdown acceleration.
        breadth_threshold: Fraction of strategies in DD that triggers warning.
    """

    def __init__(
        self,
        mc_simulations: int = 5000,
        mc_horizon: int = 252,
        velocity_window: int = 5,
        breadth_threshold: float = 0.75,
    ) -> None:
        self.mc_simulations = mc_simulations
        self.mc_horizon = mc_horizon
        self.velocity_window = velocity_window
        self.breadth_threshold = breadth_threshold

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_severity(drawdown: float) -> DrawdownSeverity:
        """Classify drawdown depth into severity bucket."""
        for sev, (lo, hi) in SEVERITY_THRESHOLDS.items():
            if lo <= drawdown < hi:
                return sev
        return DrawdownSeverity.CATASTROPHIC

    # ------------------------------------------------------------------
    # Historical episode detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_episodes(
        equity: pd.Series,
        min_drawdown: float = 0.01,
        regimes: Optional[pd.Series] = None,
    ) -> List[DrawdownEpisode]:
        """Detect all drawdown episodes from an equity curve."""
        if equity.empty:
            return []

        hwm = equity.expanding().max()
        dd = 1.0 - equity / hwm
        episodes: List[DrawdownEpisode] = []
        in_dd = False
        start_idx = 0
        peak_eq = 0.0

        for i in range(len(dd)):
            if not in_dd and dd.iloc[i] >= min_drawdown:
                in_dd = True
                start_idx = max(0, i - 1)
                peak_eq = float(hwm.iloc[i])
            elif in_dd and dd.iloc[i] < min_drawdown * 0.1:
                trough_idx = int(dd.iloc[start_idx:i + 1].idxmax()
                                 if hasattr(dd.iloc[start_idx:i + 1].idxmax(), '__index__')
                                 else dd.iloc[start_idx:i + 1].values.argmax() + start_idx)
                # Find actual trough position
                segment = dd.iloc[start_idx:i + 1]
                trough_pos = int(segment.values.argmax())
                trough_abs_idx = start_idx + trough_pos

                max_dd = float(dd.iloc[trough_abs_idx])
                if max_dd >= min_drawdown:
                    trough_dt = equity.index[trough_abs_idx]
                    regime_at_trough = None
                    if regimes is not None and trough_dt in regimes.index:
                        regime_at_trough = str(regimes.loc[trough_dt])

                    recovery_days = i - trough_abs_idx
                    episodes.append(DrawdownEpisode(
                        start_date=equity.index[start_idx],
                        trough_date=trough_dt,
                        recovery_date=equity.index[i],
                        peak_equity=peak_eq,
                        trough_equity=float(equity.iloc[trough_abs_idx]),
                        max_drawdown=max_dd,
                        severity=DrawdownPredictor.classify_severity(max_dd),
                        recovery_days=recovery_days,
                        regime_at_trough=regime_at_trough,
                    ))
                in_dd = False

        # Handle ongoing drawdown
        if in_dd:
            segment = dd.iloc[start_idx:]
            trough_pos = int(segment.values.argmax())
            trough_abs_idx = start_idx + trough_pos
            max_dd = float(dd.iloc[trough_abs_idx])
            if max_dd >= min_drawdown:
                regime_at_trough = None
                trough_dt = equity.index[trough_abs_idx]
                if regimes is not None and trough_dt in regimes.index:
                    regime_at_trough = str(regimes.loc[trough_dt])
                episodes.append(DrawdownEpisode(
                    start_date=equity.index[start_idx],
                    trough_date=trough_dt,
                    recovery_date=None,
                    peak_equity=float(hwm.iloc[start_idx]),
                    trough_equity=float(equity.iloc[trough_abs_idx]),
                    max_drawdown=max_dd,
                    severity=DrawdownPredictor.classify_severity(max_dd),
                    recovery_days=None,
                    regime_at_trough=regime_at_trough,
                ))

        return episodes

    # ------------------------------------------------------------------
    # Recovery estimation from analogues
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_recovery(
        episodes: List[DrawdownEpisode],
        current_drawdown: float,
        tolerance: float = 0.5,
    ) -> RecoveryEstimate:
        """Estimate recovery time from similar historical episodes.

        Args:
            tolerance: Fraction tolerance for analogue matching
                       (e.g. 0.5 = ±50% of current drawdown).
        """
        severity = DrawdownPredictor.classify_severity(current_drawdown)

        lo = current_drawdown * (1 - tolerance)
        hi = current_drawdown * (1 + tolerance)
        analogues = [
            e for e in episodes
            if e.recovery_days is not None and lo <= e.max_drawdown <= hi
        ]

        if not analogues:
            # Fallback: same severity bucket
            analogues = [
                e for e in episodes
                if e.recovery_days is not None and e.severity == severity
            ]

        if not analogues:
            rough = current_drawdown * TRADING_DAYS * 2
            return RecoveryEstimate(
                current_drawdown=current_drawdown, severity=severity,
                expected_days=rough, median_days=rough,
                pct_80_days=rough * 1.5, pct_95_days=rough * 2.5,
                n_analogues=0,
                size_multiplier=SEVERITY_SIZE_MULT.get(severity, 0.5),
            )

        days = np.array([e.recovery_days for e in analogues])
        return RecoveryEstimate(
            current_drawdown=current_drawdown,
            severity=severity,
            expected_days=float(days.mean()),
            median_days=float(np.median(days)),
            pct_80_days=float(np.percentile(days, 80)),
            pct_95_days=float(np.percentile(days, 95)),
            n_analogues=len(analogues),
            size_multiplier=SEVERITY_SIZE_MULT.get(severity, 0.5),
        )

    # ------------------------------------------------------------------
    # Regime-conditional recovery
    # ------------------------------------------------------------------

    @staticmethod
    def regime_recovery_profiles(
        episodes: List[DrawdownEpisode],
    ) -> List[RegimeRecoveryProfile]:
        """Recovery stats grouped by regime at trough."""
        by_regime: Dict[str, List[DrawdownEpisode]] = {}
        for e in episodes:
            if e.recovery_days is not None and e.regime_at_trough:
                by_regime.setdefault(e.regime_at_trough, []).append(e)

        results: List[RegimeRecoveryProfile] = []
        for regime, eps in sorted(by_regime.items()):
            days = [e.recovery_days for e in eps]
            dds = [e.max_drawdown for e in eps]
            results.append(RegimeRecoveryProfile(
                regime=regime, n_episodes=len(eps),
                avg_recovery_days=float(np.mean(days)),
                median_recovery_days=float(np.median(days)),
                avg_drawdown=float(np.mean(dds)),
            ))
        return results

    # ------------------------------------------------------------------
    # Monte Carlo recovery simulation
    # ------------------------------------------------------------------

    def monte_carlo_recovery(
        self,
        current_drawdown: float,
        daily_return_mean: float,
        daily_return_std: float,
        seed: int = 42,
    ) -> MonteCarloRecovery:
        """Simulate recovery paths from current drawdown.

        Models equity as geometric Brownian motion from the trough.
        Recovery = first time cumulative return exceeds drawdown.
        """
        rng = np.random.default_rng(seed)
        n = self.mc_simulations
        horizon = self.mc_horizon

        recovery_days = np.full(n, horizon, dtype=float)
        paths = np.zeros((n, horizon))

        for i in range(n):
            cum = 0.0
            for t in range(horizon):
                r = rng.normal(daily_return_mean, daily_return_std)
                cum += r
                paths[i, t] = cum
                if cum >= current_drawdown:
                    recovery_days[i] = t + 1
                    break

        # Confidence bands (10th, 50th, 90th percentile of path)
        bands: Dict[str, List[float]] = {}
        for pct_label, pct in [("p10", 10), ("p50", 50), ("p90", 90)]:
            bands[pct_label] = [float(np.percentile(paths[:, t], pct))
                                for t in range(horizon)]

        p30 = float((recovery_days <= 30).sum() / n)
        p60 = float((recovery_days <= 60).sum() / n)
        p90 = float((recovery_days <= 90).sum() / n)

        return MonteCarloRecovery(
            current_drawdown=current_drawdown,
            n_simulations=n,
            expected_days=float(recovery_days.mean()),
            median_days=float(np.median(recovery_days)),
            pct_80_days=float(np.percentile(recovery_days, 80)),
            pct_95_days=float(np.percentile(recovery_days, 95)),
            prob_recover_30d=p30,
            prob_recover_60d=p60,
            prob_recover_90d=p90,
            confidence_bands=bands,
        )

    # ------------------------------------------------------------------
    # Early warning signals
    # ------------------------------------------------------------------

    def early_warnings(
        self,
        equity: pd.Series,
        strategy_equities: Optional[Dict[str, pd.Series]] = None,
    ) -> List[EarlyWarning]:
        """Check for drawdown acceleration and breadth deterioration."""
        warnings: List[EarlyWarning] = []
        if equity.empty:
            return warnings

        hwm = equity.expanding().max()
        dd = 1.0 - equity / hwm

        # 1. Drawdown acceleration
        if len(dd) > self.velocity_window * 2:
            recent = dd.iloc[-self.velocity_window:]
            prev = dd.iloc[-2 * self.velocity_window:-self.velocity_window]
            recent_vel = float(recent.diff().mean())
            prev_vel = float(prev.diff().mean())
            accel = recent_vel - prev_vel
            threshold = 0.001
            warnings.append(EarlyWarning(
                signal_type="acceleration",
                value=accel,
                threshold=threshold,
                triggered=accel > threshold,
                message=f"DD acceleration {accel:.4f} vs threshold {threshold:.4f}",
            ))

        # 2. Drawdown velocity
        if len(dd) >= self.velocity_window:
            vel = float(dd.iloc[-1] - dd.iloc[-self.velocity_window])
            vel_thresh = 0.02
            warnings.append(EarlyWarning(
                signal_type="velocity",
                value=vel,
                threshold=vel_thresh,
                triggered=vel > vel_thresh,
                message=f"DD velocity {vel:.4f} over {self.velocity_window}d",
            ))

        # 3. Breadth deterioration
        if strategy_equities:
            n_in_dd = 0
            for name, seq in strategy_equities.items():
                if not seq.empty:
                    s_hwm = seq.expanding().max()
                    s_dd = 1.0 - seq / s_hwm
                    if float(s_dd.iloc[-1]) > 0.03:
                        n_in_dd += 1
            breadth = n_in_dd / len(strategy_equities)
            warnings.append(EarlyWarning(
                signal_type="breadth",
                value=breadth,
                threshold=self.breadth_threshold,
                triggered=breadth >= self.breadth_threshold,
                message=f"{n_in_dd}/{len(strategy_equities)} strategies in drawdown",
            ))

        return warnings

    # ------------------------------------------------------------------
    # Position sizing during drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def size_multiplier(drawdown: float) -> float:
        """Position sizing adjustment based on drawdown severity."""
        severity = DrawdownPredictor.classify_severity(drawdown)
        return SEVERITY_SIZE_MULT.get(severity, 0.5)

    # ------------------------------------------------------------------
    # Full prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        equity: pd.Series,
        current_drawdown: Optional[float] = None,
        regimes: Optional[pd.Series] = None,
        strategy_equities: Optional[Dict[str, pd.Series]] = None,
        run_mc: bool = True,
    ) -> DrawdownPrediction:
        """Run full prediction pipeline."""
        if equity.empty:
            return DrawdownPrediction(
                current_drawdown=0.0,
                severity=DrawdownSeverity.SHALLOW,
                recovery=RecoveryEstimate(0.0, DrawdownSeverity.SHALLOW,
                                           0, 0, 0, 0, 0, 1.0),
            )

        hwm = equity.expanding().max()
        dd = 1.0 - equity / hwm
        if current_drawdown is None:
            current_drawdown = float(dd.iloc[-1])

        severity = self.classify_severity(current_drawdown)
        episodes = self.detect_episodes(equity, regimes=regimes)
        recovery = self.estimate_recovery(episodes, current_drawdown)
        regime_profiles = self.regime_recovery_profiles(episodes)
        warnings = self.early_warnings(equity, strategy_equities)

        mc = None
        if run_mc and current_drawdown > 0.005:
            returns = equity.pct_change().dropna()
            mu = float(returns.mean())
            sigma = float(returns.std())
            mc = self.monte_carlo_recovery(current_drawdown, mu, sigma)

        return DrawdownPrediction(
            current_drawdown=current_drawdown,
            severity=severity,
            recovery=recovery,
            monte_carlo=mc,
            warnings=warnings,
            regime_profiles=regime_profiles,
            size_multiplier=self.size_multiplier(current_drawdown),
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 720, height: int = 210, color: str = "#2980b9",
        bands: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        all_v = list(values)
        if bands:
            for bv in bands.values():
                all_v.extend(bv[:n])
        vmin = min(all_v)
        vmax = max(all_v)
        if vmax <= vmin:
            vmax = vmin + 0.01
        pad_l, pad_r, pad_t, pad_b = 55, 15, 28, 25
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

        def tx(i: int) -> float:
            return pad_l + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return pad_t + (1 - (v - vmin) / (vmax - vmin)) * ph

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')

        if bands:
            band_colors = {"p10": "#dfe6e9", "p50": "#b2bec3", "p90": "#dfe6e9"}
            for label, bv in bands.items():
                bc = band_colors.get(label, "#eee")
                d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(bv[i]):.1f}"
                              for i in range(min(n, len(bv))))
                p.append(f'<path d="{d}" fill="none" stroke="{bc}" stroke-width="1"/>')

        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                      for i, v in enumerate(values))
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_waterfall(episodes: List[DrawdownEpisode], width: int = 720, height: int = 160) -> str:
        if not episodes:
            return ""
        n = len(episodes)
        colors = {
            DrawdownSeverity.SHALLOW: "#27ae60",
            DrawdownSeverity.MODERATE: "#f1c40f",
            DrawdownSeverity.DEEP: "#e67e22",
            DrawdownSeverity.CATASTROPHIC: "#e74c3c",
        }
        pad_l = 50
        bw = (width - pad_l - 20) / max(n, 1) * 0.8
        gap = (width - pad_l - 20) / max(n, 1)
        max_dd = max(e.max_drawdown for e in episodes)
        if max_dd <= 0:
            max_dd = 0.01
        ph = height - 50

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">Drawdown Episodes</text>')
        for i, e in enumerate(episodes):
            x = pad_l + i * gap + (gap - bw) / 2
            bh = e.max_drawdown / max_dd * ph
            y = 25 + ph - bh
            c = colors.get(e.severity, "#999")
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" '
                     f'height="{bh:.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" '
                     f'font-size="8" fill="#333">{e.max_drawdown:.1%}</text>')
        # legend
        lx = pad_l
        for sev, c in colors.items():
            p.append(f'<rect x="{lx}" y="{height - 14}" width="8" height="8" fill="{c}"/>')
            p.append(f'<text x="{lx + 11}" y="{height - 6}" font-size="8" fill="#333">{sev.value}</text>')
            lx += 90
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        prediction: DrawdownPrediction,
        equity: Optional[pd.Series] = None,
        output_path: str = "reports/drawdown_predictor.html",
    ) -> str:
        """HTML report: waterfall, recovery timeline, MC bands, regime comparison."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # DD path
        dd_svg = ""
        if equity is not None and not equity.empty:
            hwm = equity.expanding().max()
            dd = (1.0 - equity / hwm).tolist()
            dd_svg = self._svg_line(dd, "Drawdown Path", color="#e74c3c")

        # MC confidence bands
        mc_svg = ""
        if prediction.monte_carlo and prediction.monte_carlo.confidence_bands:
            mc = prediction.monte_carlo
            p50 = mc.confidence_bands.get("p50", [])
            if len(p50) > 2:
                mc_svg = self._svg_line(
                    p50[:min(120, len(p50))],
                    "Monte Carlo Recovery (median path)",
                    color="#2980b9",
                    bands={k: v[:min(120, len(v))]
                           for k, v in mc.confidence_bands.items() if k != "p50"},
                )

        # Episode waterfall
        episodes = self.detect_episodes(equity) if equity is not None and not equity.empty else []
        waterfall_svg = self._svg_waterfall(episodes)

        # MC table
        mc_html = ""
        if prediction.monte_carlo:
            mc = prediction.monte_carlo
            mc_html = f"""
<h2>Monte Carlo Recovery ({mc.n_simulations:,} sims)</h2>
<table class="m"><tr><th>Expected</th><th>Median</th><th>80th pct</th>
<th>95th pct</th><th>P(30d)</th><th>P(60d)</th><th>P(90d)</th></tr>
<tr><td>{mc.expected_days:.0f}d</td><td>{mc.median_days:.0f}d</td>
<td>{mc.pct_80_days:.0f}d</td><td>{mc.pct_95_days:.0f}d</td>
<td>{mc.prob_recover_30d:.1%}</td><td>{mc.prob_recover_60d:.1%}</td>
<td>{mc.prob_recover_90d:.1%}</td></tr></table>
{mc_svg}"""

        # Regime profiles
        rp_html = ""
        if prediction.regime_profiles:
            rows = [
                f"<tr><td>{rp.regime}</td><td>{rp.n_episodes}</td>"
                f"<td>{rp.avg_recovery_days:.0f}</td>"
                f"<td>{rp.median_recovery_days:.0f}</td>"
                f"<td>{rp.avg_drawdown:.2%}</td></tr>"
                for rp in prediction.regime_profiles
            ]
            rp_html = f"""
<h2>Regime-Conditional Recovery</h2>
<table><tr><th>Regime</th><th>Episodes</th><th>Avg Days</th>
<th>Median Days</th><th>Avg DD</th></tr>
{''.join(rows)}</table>"""

        # Warnings
        warn_html = ""
        if prediction.warnings:
            rows = [
                f"<tr><td>{w.signal_type}</td><td>{w.value:.4f}</td>"
                f"<td>{w.threshold:.4f}</td>"
                f"<td class='{'warn' if w.triggered else ''}'>"
                f"{'YES' if w.triggered else 'no'}</td>"
                f"<td>{w.message}</td></tr>"
                for w in prediction.warnings
            ]
            warn_html = f"""
<h2>Early Warning Signals</h2>
<table><tr><th>Signal</th><th>Value</th><th>Threshold</th>
<th>Triggered</th><th>Message</th></tr>
{''.join(rows)}</table>"""

        sev_colors = {
            "shallow": "#27ae60", "moderate": "#f1c40f",
            "deep": "#e67e22", "catastrophic": "#e74c3c",
        }
        sc = sev_colors.get(prediction.severity.value, "#999")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Drawdown Predictor</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.m {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
td.warn {{ color: #e74c3c; font-weight: bold; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.badge {{ display: inline-block; padding: 3px 12px; border-radius: 8px;
          color: #fff; font-weight: bold; }}
</style></head><body>
<h1>Drawdown Recovery Predictor</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Current Drawdown:</strong> {prediction.current_drawdown:.2%}
   — <span class="badge" style="background:{sc}">{prediction.severity.value.upper()}</span></p>
<p><strong>Expected Recovery:</strong> {prediction.recovery.expected_days:.0f} days
   (median {prediction.recovery.median_days:.0f}d,
    80th pct {prediction.recovery.pct_80_days:.0f}d) |
   Analogues: {prediction.recovery.n_analogues}</p>
<p><strong>Size Multiplier:</strong> {prediction.size_multiplier:.0%}</p>
</div>

<h2>Drawdown Path</h2>
{dd_svg}

<h2>Historical Episodes</h2>
{waterfall_svg}

{mc_html}
{rp_html}
{warn_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Drawdown predictor report -> %s", path)
        return str(path)
