"""Dynamic risk limit engine — adaptive limits based on regime, volatility,
and drawdown state with real-time monitoring, breach classification,
automatic reduction triggers, and stress-adjusted tightening.

Provides:
  1. Adaptive position limits (regime + vol + drawdown)
  2. Exposure limits (gross, net, sector, beta)
  3. Real-time monitoring with breach severity (info/warning/critical)
  4. Automatic position reduction triggers
  5. Limit history and utilisation tracking
  6. Stress-adjusted limits (tighten in crisis)
  7. HTML report with gauges, breach history, exposure heatmap
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Severity ────────────────────────────────────────────────────────────────
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"

# ── Regimes ─────────────────────────────────────────────────────────────────
BULL = "bull"
BEAR = "bear"
HIGH_VOL = "high_vol"
LOW_VOL = "low_vol"
CRASH = "crash"


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class LimitConfig:
    """Base limit thresholds (pre-adjustment)."""
    max_gross_exposure: float = 1.0        # fraction of NAV
    max_net_exposure: float = 0.50
    max_single_position: float = 0.10
    max_sector_exposure: float = 0.30
    max_beta_exposure: float = 1.5
    max_drawdown_pct: float = 0.15
    max_positions: int = 20
    # Breach thresholds (fraction of limit)
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95


@dataclass
class RegimeMultipliers:
    """Limit multipliers per regime (< 1 = tighten)."""
    gross: float = 1.0
    net: float = 1.0
    single: float = 1.0
    positions: float = 1.0


DEFAULT_REGIME_MULTIPLIERS: Dict[str, RegimeMultipliers] = {
    BULL: RegimeMultipliers(1.0, 1.0, 1.0, 1.0),
    LOW_VOL: RegimeMultipliers(1.1, 1.1, 1.1, 1.1),
    BEAR: RegimeMultipliers(0.70, 0.60, 0.70, 0.80),
    HIGH_VOL: RegimeMultipliers(0.50, 0.40, 0.50, 0.60),
    CRASH: RegimeMultipliers(0.25, 0.20, 0.25, 0.30),
}


@dataclass
class EffectiveLimit:
    """Limit after regime / vol / drawdown adjustment."""
    limit_name: str
    base_value: float
    adjusted_value: float
    current_value: float
    utilisation: float           # current / adjusted (0–1+)
    headroom: float              # adjusted - current
    regime_mult: float = 1.0
    vol_mult: float = 1.0
    dd_mult: float = 1.0


@dataclass
class Breach:
    """A limit breach event."""
    timestamp: str
    limit_name: str
    severity: str               # info / warning / critical
    current: float
    limit: float
    utilisation: float
    detail: str = ""


@dataclass
class ReductionTrigger:
    """Automatic position reduction recommendation."""
    trigger_type: str           # "drawdown", "exposure", "vol"
    severity: str
    current_value: float
    threshold: float
    recommended_action: str
    reduction_pct: float        # how much to reduce (0–1)


@dataclass
class ExposureSnapshot:
    """Current portfolio exposure state."""
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    beta_exposure: float = 0.0
    n_positions: int = 0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    position_sizes: Dict[str, float] = field(default_factory=dict)


@dataclass
class LimitHistoryEntry:
    """One point in the limit utilisation history."""
    timestamp: str
    limit_name: str
    utilisation: float
    adjusted_limit: float


@dataclass
class RiskLimitResult:
    """Complete risk limit monitoring output."""
    effective_limits: List[EffectiveLimit] = field(default_factory=list)
    breaches: List[Breach] = field(default_factory=list)
    triggers: List[ReductionTrigger] = field(default_factory=list)
    exposure: Optional[ExposureSnapshot] = None
    history: List[LimitHistoryEntry] = field(default_factory=list)
    regime: str = ""
    vix: float = 0.0
    current_dd: float = 0.0
    n_breaches: int = 0
    n_critical: int = 0
    generated_at: str = ""


# ── Core engine ─────────────────────────────────────────────────────────────
class RiskLimitEngine:
    """Dynamic risk limit engine with adaptive tightening."""

    def __init__(
        self,
        config: Optional[LimitConfig] = None,
        regime_multipliers: Optional[Dict[str, RegimeMultipliers]] = None,
        vix_tighten_start: float = 25.0,
        vix_tighten_full: float = 50.0,
        dd_tighten_start: float = 0.05,
        dd_tighten_full: float = 0.15,
    ) -> None:
        self.config = config or LimitConfig()
        self.regime_mults = regime_multipliers or dict(DEFAULT_REGIME_MULTIPLIERS)
        self.vix_start = vix_tighten_start
        self.vix_full = vix_tighten_full
        self.dd_start = dd_tighten_start
        self.dd_full = dd_tighten_full
        self._history: List[LimitHistoryEntry] = []

    # ── Public API ──────────────────────────────────────────────────────────
    def monitor(
        self,
        exposure: ExposureSnapshot,
        regime: str = BULL,
        vix: float = 15.0,
        current_dd: float = 0.0,
    ) -> RiskLimitResult:
        """Check all limits against current exposure.

        Parameters
        ----------
        exposure : ExposureSnapshot
            Current portfolio exposure state.
        regime : str
            Current market regime.
        vix : float
            Current VIX level.
        current_dd : float
            Current drawdown depth (positive fraction, e.g. 0.10 = 10%).
        """
        regime_m = self.regime_mults.get(regime, RegimeMultipliers())
        vol_m = self._vol_multiplier(vix)
        dd_m = self._dd_multiplier(current_dd)

        limits = self._compute_limits(exposure, regime_m, vol_m, dd_m)
        breaches = self._detect_breaches(limits)
        triggers = self._compute_triggers(exposure, limits, current_dd, vix)

        # Record history
        now = self._now()
        for lim in limits:
            self._history.append(LimitHistoryEntry(
                timestamp=now, limit_name=lim.limit_name,
                utilisation=lim.utilisation, adjusted_limit=lim.adjusted_value,
            ))

        n_crit = sum(1 for b in breaches if b.severity == CRITICAL)

        return RiskLimitResult(
            effective_limits=limits,
            breaches=breaches,
            triggers=triggers,
            exposure=exposure,
            history=list(self._history),
            regime=regime,
            vix=vix,
            current_dd=current_dd,
            n_breaches=len(breaches),
            n_critical=n_crit,
            generated_at=now,
        )

    def get_history(self) -> List[LimitHistoryEntry]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def generate_report(
        self,
        result: RiskLimitResult,
        output_path: str | Path = "reports/risk_limits.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Risk limits report written to %s", path)
        return path

    # ── Multipliers ─────────────────────────────────────────────────────────
    def _vol_multiplier(self, vix: float) -> float:
        """Linear tightening from 1.0 at vix_start to 0.5 at vix_full."""
        if vix <= self.vix_start:
            return 1.0
        if vix >= self.vix_full:
            return 0.5
        frac = (vix - self.vix_start) / (self.vix_full - self.vix_start)
        return 1.0 - frac * 0.5

    def _dd_multiplier(self, dd: float) -> float:
        """Linear tightening from 1.0 at dd_start to 0.4 at dd_full."""
        if dd <= self.dd_start:
            return 1.0
        if dd >= self.dd_full:
            return 0.4
        frac = (dd - self.dd_start) / (self.dd_full - self.dd_start)
        return 1.0 - frac * 0.6

    # ── Limit computation ───────────────────────────────────────────────────
    def _compute_limits(
        self,
        exp: ExposureSnapshot,
        rm: RegimeMultipliers,
        vol_m: float,
        dd_m: float,
    ) -> List[EffectiveLimit]:
        cfg = self.config
        limits: List[EffectiveLimit] = []

        def _lim(name: str, base: float, current: float, r_mult: float) -> EffectiveLimit:
            adj = base * r_mult * vol_m * dd_m
            util = current / adj if adj > 1e-12 else 0.0
            return EffectiveLimit(
                limit_name=name, base_value=base, adjusted_value=adj,
                current_value=current, utilisation=util,
                headroom=adj - current,
                regime_mult=r_mult, vol_mult=vol_m, dd_mult=dd_m,
            )

        limits.append(_lim("gross_exposure", cfg.max_gross_exposure, exp.gross_exposure, rm.gross))
        limits.append(_lim("net_exposure", cfg.max_net_exposure, abs(exp.net_exposure), rm.net))
        limits.append(_lim("beta_exposure", cfg.max_beta_exposure, abs(exp.beta_exposure), 1.0))
        limits.append(_lim("max_positions", float(cfg.max_positions), float(exp.n_positions), rm.positions))

        # Largest single position
        max_pos = max(abs(v) for v in exp.position_sizes.values()) if exp.position_sizes else 0.0
        limits.append(_lim("single_position", cfg.max_single_position, max_pos, rm.single))

        # Sector limits
        for sector, size in exp.sector_exposures.items():
            limits.append(_lim(
                f"sector_{sector}", cfg.max_sector_exposure, abs(size), 1.0,
            ))

        return limits

    # ── Breach detection ────────────────────────────────────────────────────
    def _detect_breaches(self, limits: List[EffectiveLimit]) -> List[Breach]:
        breaches: List[Breach] = []
        now = self._now()
        for lim in limits:
            severity = self._classify_breach(lim.utilisation)
            if severity is not None:
                breaches.append(Breach(
                    timestamp=now,
                    limit_name=lim.limit_name,
                    severity=severity,
                    current=lim.current_value,
                    limit=lim.adjusted_value,
                    utilisation=lim.utilisation,
                    detail=f"{lim.limit_name}: {lim.current_value:.2f} / {lim.adjusted_value:.2f}",
                ))
        return breaches

    def _classify_breach(self, utilisation: float) -> Optional[str]:
        if utilisation >= self.config.critical_threshold:
            return CRITICAL
        if utilisation >= self.config.warning_threshold:
            return WARNING
        return None

    # ── Reduction triggers ──────────────────────────────────────────────────
    def _compute_triggers(
        self,
        exp: ExposureSnapshot,
        limits: List[EffectiveLimit],
        dd: float,
        vix: float,
    ) -> List[ReductionTrigger]:
        triggers: List[ReductionTrigger] = []

        # Drawdown trigger
        if dd >= self.dd_start:
            pct = min(1.0, (dd - self.dd_start) / max(self.dd_full - self.dd_start, 0.01))
            reduction = 0.10 + pct * 0.40  # 10% to 50%
            severity = CRITICAL if dd >= self.dd_full else WARNING
            triggers.append(ReductionTrigger(
                trigger_type="drawdown", severity=severity,
                current_value=dd, threshold=self.dd_start,
                recommended_action=f"Reduce positions by {reduction:.0%}",
                reduction_pct=reduction,
            ))

        # VIX trigger
        if vix >= self.vix_start:
            pct = min(1.0, (vix - self.vix_start) / max(self.vix_full - self.vix_start, 1))
            reduction = 0.10 + pct * 0.30
            severity = CRITICAL if vix >= self.vix_full else WARNING
            triggers.append(ReductionTrigger(
                trigger_type="vol", severity=severity,
                current_value=vix, threshold=self.vix_start,
                recommended_action=f"Reduce gross exposure by {reduction:.0%}",
                reduction_pct=reduction,
            ))

        # Over-limit triggers
        for lim in limits:
            if lim.utilisation >= 1.0:
                excess = lim.current_value - lim.adjusted_value
                reduction = min(0.50, excess / max(lim.current_value, 1e-9))
                triggers.append(ReductionTrigger(
                    trigger_type="exposure", severity=CRITICAL,
                    current_value=lim.current_value, threshold=lim.adjusted_value,
                    recommended_action=f"Reduce {lim.limit_name} by {reduction:.0%}",
                    reduction_pct=reduction,
                ))

        return triggers

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: RiskLimitResult) -> str:
        cards = self._html_cards(r)
        gauges = self._svg_gauges(r.effective_limits)
        breach_tbl = self._html_breaches(r.breaches)
        trigger_tbl = self._html_triggers(r.triggers)
        exposure_tbl = self._html_exposure(r.exposure)
        limits_tbl = self._html_limits(r.effective_limits)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Risk Limits</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:28px}}
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
.gauge-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:28px}}
</style>
</head>
<body>
<h1>Risk Limit Monitor</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; Regime: <strong>{r.regime.upper()}</strong> &middot; VIX: {r.vix:.1f} &middot; DD: {r.current_dd:.1%}</p>
{cards}
<div class="sec"><h2>Limit Utilisation</h2><div class="gauge-grid">{gauges}</div></div>
{limits_tbl}
{breach_tbl}
{trigger_tbl}
{exposure_tbl}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: RiskLimitResult) -> str:
        max_util = max((l.utilisation for l in r.effective_limits), default=0)
        util_cls = "neg" if max_util >= 0.95 else "warn" if max_util >= 0.80 else "pos"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Breaches</div><div class="val {'neg' if r.n_breaches else ''}">{r.n_breaches}</div></div>
<div class="card"><div class="lbl">Critical</div><div class="val {'neg' if r.n_critical else ''}">{r.n_critical}</div></div>
<div class="card"><div class="lbl">Max Utilisation</div><div class="val {util_cls}">{max_util:.0%}</div></div>
<div class="card"><div class="lbl">Triggers</div><div class="val">{len(r.triggers)}</div></div>
</div>"""

    @staticmethod
    def _svg_gauges(limits: List[EffectiveLimit]) -> str:
        if not limits:
            return ""
        # Filter to main limits (not per-sector)
        main = [l for l in limits if not l.limit_name.startswith("sector_")][:6]
        gauges = ""
        for l in main:
            util = min(l.utilisation, 1.2)
            pct = min(util * 100, 100)
            colour = "#4ade80" if util < 0.80 else "#fbbf24" if util < 0.95 else "#f87171"
            w, h = 130, 80
            # Arc gauge
            angle = util * 180
            angle = min(angle, 180)
            rad = angle * 3.14159 / 180
            cx, cy, r = 65, 60, 45
            ex = cx + r * (-np.cos(rad))  # type: ignore
            ey = cy - r * np.sin(rad)
            large = 1 if angle > 90 else 0
            arc = f"M {cx - r} {cy} A {r} {r} 0 {large} 1 {ex:.0f} {ey:.0f}"
            gauges += (
                f'<div style="text-align:center">'
                f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
                f'<path d="M {cx - r} {cy} A {r} {r} 0 1 1 {cx + r} {cy}" fill="none" stroke="#334155" stroke-width="8" stroke-linecap="round"/>'
                f'<path d="{arc}" fill="none" stroke="{colour}" stroke-width="8" stroke-linecap="round"/>'
                f'<text x="{cx}" y="{cy - 5}" text-anchor="middle" font-size="16" font-weight="700" fill="{colour}">{util:.0%}</text>'
                f'</svg>'
                f'<div style="font-size:.75rem;color:#94a3b8">{l.limit_name.replace("_"," ")}</div>'
                f'</div>'
            )
        return gauges

    @staticmethod
    def _html_limits(limits: List[EffectiveLimit]) -> str:
        if not limits:
            return ""
        rows = ""
        for l in limits:
            cls = "neg" if l.utilisation >= 0.95 else "warn" if l.utilisation >= 0.80 else ""
            rows += (f"<tr><td>{l.limit_name}</td><td>{l.base_value:.2f}</td>"
                     f"<td>{l.adjusted_value:.2f}</td><td>{l.current_value:.2f}</td>"
                     f'<td class="{cls}">{l.utilisation:.0%}</td>'
                     f"<td>{l.regime_mult:.2f}</td><td>{l.vol_mult:.2f}</td><td>{l.dd_mult:.2f}</td></tr>")
        return f"""<div class="sec"><h2>Effective Limits</h2>
<table><thead><tr><th>Limit</th><th>Base</th><th>Adjusted</th><th>Current</th><th>Util</th><th>Regime</th><th>Vol</th><th>DD</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_breaches(breaches: List[Breach]) -> str:
        if not breaches:
            return ""
        rows = ""
        for b in breaches:
            cls = "neg" if b.severity == CRITICAL else "warn"
            rows += (f'<tr><td class="{cls}">{b.severity.upper()}</td>'
                     f"<td>{b.limit_name}</td><td>{b.current:.2f}</td>"
                     f"<td>{b.limit:.2f}</td><td>{b.utilisation:.0%}</td>"
                     f"<td>{b.timestamp}</td></tr>")
        return f"""<div class="sec"><h2>Breach History</h2>
<table><thead><tr><th>Severity</th><th>Limit</th><th>Current</th><th>Limit</th><th>Util</th><th>Time</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_triggers(triggers: List[ReductionTrigger]) -> str:
        if not triggers:
            return ""
        rows = ""
        for t in triggers:
            cls = "neg" if t.severity == CRITICAL else "warn"
            rows += (f'<tr><td class="{cls}">{t.severity.upper()}</td>'
                     f"<td>{t.trigger_type}</td><td>{t.current_value:.2f}</td>"
                     f"<td>{t.threshold:.2f}</td><td>{t.reduction_pct:.0%}</td>"
                     f"<td>{t.recommended_action}</td></tr>")
        return f"""<div class="sec"><h2>Reduction Triggers</h2>
<table><thead><tr><th>Severity</th><th>Type</th><th>Current</th><th>Threshold</th><th>Reduction</th><th>Action</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_exposure(exp: Optional[ExposureSnapshot]) -> str:
        if not exp:
            return ""
        sector_rows = "".join(
            f"<tr><td>{s}</td><td>{v:.2f}</td></tr>"
            for s, v in sorted(exp.sector_exposures.items())
        )
        return f"""<div class="sec"><h2>Exposure Snapshot</h2>
<table><tbody>
<tr><td>Gross</td><td>{exp.gross_exposure:.2f}</td></tr>
<tr><td>Net</td><td>{exp.net_exposure:.2f}</td></tr>
<tr><td>Beta</td><td>{exp.beta_exposure:.2f}</td></tr>
<tr><td>Positions</td><td>{exp.n_positions}</td></tr>
</tbody></table>
{'<h3 style="margin-top:12px;font-size:.95rem;color:#94a3b8">Sector Exposures</h3><table><thead><tr><th>Sector</th><th>Exposure</th></tr></thead><tbody>' + sector_rows + '</tbody></table>' if sector_rows else ''}
</div>"""
