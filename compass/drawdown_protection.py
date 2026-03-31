"""
Dynamic drawdown protection system.

Graduated response levels tied to drawdown depth from high-water mark:
  GREEN   < 3%   — normal operations
  YELLOW  3-5%   — reduce new position sizes by 50%
  ORANGE  5-8%   — hedge existing positions, halt new entries
  RED     > 8%   — flatten portfolio, full halt

Additional dimensions:
  - Per-strategy drawdown tracking (independent limits)
  - Drawdown velocity (rate of drawdown acceleration)
  - Recovery estimation (expected days to HWM from historical stats)
  - Correlation-conditional protection (tighten when corr spikes)
  - Automated action recommendations per level

All methods are pure computation — no broker connections.
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
# Enums & thresholds
# ---------------------------------------------------------------------------

class ProtectionLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


DEFAULT_THRESHOLDS: Dict[ProtectionLevel, float] = {
    ProtectionLevel.GREEN: 0.0,
    ProtectionLevel.YELLOW: 0.03,
    ProtectionLevel.ORANGE: 0.05,
    ProtectionLevel.RED: 0.08,
}

LEVEL_ACTIONS: Dict[ProtectionLevel, str] = {
    ProtectionLevel.GREEN: "normal",
    ProtectionLevel.YELLOW: "reduce_size",
    ProtectionLevel.ORANGE: "hedge_and_halt",
    ProtectionLevel.RED: "flatten_all",
}

LEVEL_SIZE_MULT: Dict[ProtectionLevel, float] = {
    ProtectionLevel.GREEN: 1.0,
    ProtectionLevel.YELLOW: 0.50,
    ProtectionLevel.ORANGE: 0.0,
    ProtectionLevel.RED: 0.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DrawdownState:
    """Current drawdown snapshot."""
    date: datetime
    equity: float
    high_water_mark: float
    drawdown: float              # 0..1 (fraction below HWM)
    level: ProtectionLevel
    action: str
    size_multiplier: float


@dataclass
class StrategyDrawdown:
    """Per-strategy drawdown."""
    name: str
    equity: float
    hwm: float
    drawdown: float
    level: ProtectionLevel


@dataclass
class DrawdownVelocity:
    """Rate of drawdown change."""
    date: datetime
    velocity_1d: float           # 1-day change in drawdown
    velocity_5d: float           # 5-day annualised
    is_accelerating: bool


@dataclass
class RecoveryEstimate:
    """Expected recovery to HWM."""
    current_drawdown: float
    expected_days: float
    confidence_80: float         # 80th percentile days
    historical_avg_recovery: float
    n_historical_episodes: int


@dataclass
class CorrelationProtection:
    """Correlation-conditional tightening."""
    avg_correlation: float
    correlation_percentile: float
    threshold_multiplier: float  # <1 = tighter thresholds
    adjusted_thresholds: Dict[ProtectionLevel, float] = field(default_factory=dict)


@dataclass
class ProtectionEvent:
    """Record of a level change."""
    date: datetime
    old_level: ProtectionLevel
    new_level: ProtectionLevel
    drawdown: float
    action: str
    trigger: str                 # "drawdown" | "velocity" | "correlation"


@dataclass
class ProtectionEffectiveness:
    """How well protection limited losses."""
    max_drawdown_with: float
    max_drawdown_without: float
    reduction_pct: float
    n_interventions: int
    avg_recovery_days: float


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class DrawdownProtection:
    """Dynamic drawdown protection with graduated responses.

    Args:
        thresholds: {ProtectionLevel: drawdown_fraction} overrides.
        velocity_lookback: Days for velocity calculation.
        velocity_threshold: Annualised velocity that triggers escalation.
        correlation_lookback: Days for rolling correlation.
        correlation_escalation: Corr percentile above which thresholds tighten.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[ProtectionLevel, float]] = None,
        velocity_lookback: int = 5,
        velocity_threshold: float = 0.50,
        correlation_lookback: int = 21,
        correlation_escalation: float = 0.80,
    ) -> None:
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.velocity_lookback = velocity_lookback
        self.velocity_threshold = velocity_threshold
        self.correlation_lookback = correlation_lookback
        self.correlation_escalation = correlation_escalation

        self._hwm: float = 0.0
        self._current_level: ProtectionLevel = ProtectionLevel.GREEN
        self._events: List[ProtectionEvent] = []
        self._state_history: List[DrawdownState] = []
        self._strategy_hwms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core drawdown
    # ------------------------------------------------------------------

    def classify_level(self, drawdown: float) -> ProtectionLevel:
        """Map drawdown fraction to protection level."""
        level = ProtectionLevel.GREEN
        for lvl in [ProtectionLevel.RED, ProtectionLevel.ORANGE,
                     ProtectionLevel.YELLOW]:
            if drawdown >= self.thresholds[lvl]:
                level = lvl
                break
        return level

    def update(
        self, equity: float, date: Optional[datetime] = None,
    ) -> DrawdownState:
        """Update with new equity value, return current state."""
        dt = date or datetime.now()
        if equity > self._hwm:
            self._hwm = equity

        dd = 1.0 - equity / self._hwm if self._hwm > 0 else 0.0
        dd = max(dd, 0.0)
        level = self.classify_level(dd)
        action = LEVEL_ACTIONS[level]
        size_mult = LEVEL_SIZE_MULT[level]

        # Record level change
        if level != self._current_level:
            self._events.append(ProtectionEvent(
                date=dt, old_level=self._current_level,
                new_level=level, drawdown=dd,
                action=action, trigger="drawdown",
            ))
            self._current_level = level

        state = DrawdownState(
            date=dt, equity=equity, high_water_mark=self._hwm,
            drawdown=dd, level=level, action=action,
            size_multiplier=size_mult,
        )
        self._state_history.append(state)
        return state

    def update_series(self, equity_series: pd.Series) -> List[DrawdownState]:
        """Process a full equity curve."""
        results: List[DrawdownState] = []
        for dt, eq in equity_series.items():
            results.append(self.update(float(eq), date=dt))
        return results

    # ------------------------------------------------------------------
    # Per-strategy tracking
    # ------------------------------------------------------------------

    def update_strategy(
        self, name: str, equity: float,
    ) -> StrategyDrawdown:
        """Track drawdown for an individual strategy."""
        hwm = self._strategy_hwms.get(name, 0.0)
        if equity > hwm:
            hwm = equity
            self._strategy_hwms[name] = hwm
        dd = 1.0 - equity / hwm if hwm > 0 else 0.0
        dd = max(dd, 0.0)
        level = self.classify_level(dd)
        return StrategyDrawdown(
            name=name, equity=equity, hwm=hwm,
            drawdown=dd, level=level,
        )

    def update_strategies(
        self, equities: Dict[str, float],
    ) -> List[StrategyDrawdown]:
        """Update all strategies at once."""
        return [self.update_strategy(n, eq) for n, eq in equities.items()]

    # ------------------------------------------------------------------
    # Drawdown velocity
    # ------------------------------------------------------------------

    def compute_velocity(
        self, date: Optional[datetime] = None,
    ) -> DrawdownVelocity:
        """How fast is drawdown increasing?"""
        dt = date or datetime.now()
        n = len(self._state_history)
        if n < 2:
            return DrawdownVelocity(
                date=dt, velocity_1d=0.0, velocity_5d=0.0,
                is_accelerating=False,
            )

        v1d = self._state_history[-1].drawdown - self._state_history[-2].drawdown

        lb = min(self.velocity_lookback, n)
        dd_now = self._state_history[-1].drawdown
        dd_prev = self._state_history[-lb].drawdown
        v5d = (dd_now - dd_prev) / lb * TRADING_DAYS if lb > 0 else 0.0

        # Acceleration: is velocity increasing?
        accel = False
        if n >= 2 * lb:
            dd_mid = self._state_history[-lb].drawdown
            dd_far = self._state_history[-2 * lb].drawdown
            prev_vel = (dd_mid - dd_far) / lb
            curr_vel = (dd_now - dd_mid) / lb
            accel = curr_vel > prev_vel + 1e-6

        return DrawdownVelocity(
            date=dt, velocity_1d=v1d, velocity_5d=v5d,
            is_accelerating=accel,
        )

    def check_velocity_escalation(self) -> bool:
        """True if velocity warrants level escalation."""
        vel = self.compute_velocity()
        return vel.velocity_5d > self.velocity_threshold

    # ------------------------------------------------------------------
    # Recovery estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_recovery(
        equity_series: pd.Series,
        current_drawdown: float,
    ) -> RecoveryEstimate:
        """Estimate days to recover to HWM from historical drawdown episodes."""
        if equity_series.empty or current_drawdown <= 0:
            return RecoveryEstimate(
                current_drawdown=current_drawdown, expected_days=0.0,
                confidence_80=0.0, historical_avg_recovery=0.0,
                n_historical_episodes=0,
            )

        hwm = equity_series.expanding().max()
        dd = 1.0 - equity_series / hwm

        # Find historical drawdown episodes of similar magnitude
        in_dd = dd > 0.005  # threshold for "in drawdown"
        episodes: List[int] = []
        episode_start = None

        for i in range(len(dd)):
            if in_dd.iloc[i] and episode_start is None:
                episode_start = i
            elif not in_dd.iloc[i] and episode_start is not None:
                length = i - episode_start
                peak_dd = float(dd.iloc[episode_start:i].max())
                if peak_dd >= current_drawdown * 0.5:
                    episodes.append(length)
                episode_start = None

        if not episodes:
            # Rough estimate: linear extrapolation
            return RecoveryEstimate(
                current_drawdown=current_drawdown,
                expected_days=current_drawdown * TRADING_DAYS,
                confidence_80=current_drawdown * TRADING_DAYS * 2,
                historical_avg_recovery=0.0,
                n_historical_episodes=0,
            )

        arr = np.array(episodes)
        return RecoveryEstimate(
            current_drawdown=current_drawdown,
            expected_days=float(arr.mean()),
            confidence_80=float(np.percentile(arr, 80)),
            historical_avg_recovery=float(arr.mean()),
            n_historical_episodes=len(episodes),
        )

    # ------------------------------------------------------------------
    # Correlation-conditional protection
    # ------------------------------------------------------------------

    def correlation_protection(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> CorrelationProtection:
        """Tighten thresholds when cross-strategy correlation spikes."""
        if len(strategy_returns) < 2:
            return CorrelationProtection(
                avg_correlation=0.0, correlation_percentile=0.0,
                threshold_multiplier=1.0,
                adjusted_thresholds=dict(self.thresholds),
            )

        ret_df = pd.DataFrame(strategy_returns).dropna()
        if len(ret_df) < 10:
            return CorrelationProtection(
                avg_correlation=0.0, correlation_percentile=0.0,
                threshold_multiplier=1.0,
                adjusted_thresholds=dict(self.thresholds),
            )

        corr_matrix = ret_df.corr()
        n = len(corr_matrix)
        upper = []
        for i in range(n):
            for j in range(i + 1, n):
                upper.append(corr_matrix.iloc[i, j])

        if not upper:
            return CorrelationProtection(
                avg_correlation=0.0, correlation_percentile=0.0,
                threshold_multiplier=1.0,
                adjusted_thresholds=dict(self.thresholds),
            )

        avg_corr = float(np.mean(upper))

        # Rolling correlation percentile
        lb = min(self.correlation_lookback, len(ret_df))
        recent = ret_df.iloc[-lb:]
        recent_corr = recent.corr()
        recent_upper = []
        for i in range(len(recent_corr)):
            for j in range(i + 1, len(recent_corr)):
                recent_upper.append(recent_corr.iloc[i, j])
        recent_avg = float(np.mean(recent_upper)) if recent_upper else 0.0

        # Percentile: where does recent_avg fall in the full distribution?
        pctile = float((np.array(upper) <= recent_avg).mean())

        # Tighten if above escalation threshold
        mult = 1.0
        if pctile >= self.correlation_escalation:
            mult = 0.7  # 30% tighter thresholds
        elif pctile >= self.correlation_escalation * 0.75:
            mult = 0.85

        adjusted = {lvl: th * mult for lvl, th in self.thresholds.items()}

        return CorrelationProtection(
            avg_correlation=avg_corr,
            correlation_percentile=pctile,
            threshold_multiplier=mult,
            adjusted_thresholds=adjusted,
        )

    def apply_correlation_adjustment(
        self, corr_prot: CorrelationProtection,
    ) -> None:
        """Apply tightened thresholds from correlation analysis."""
        if corr_prot.threshold_multiplier < 1.0:
            self.thresholds = corr_prot.adjusted_thresholds
            logger.info(
                "Correlation-adjusted thresholds (mult=%.2f): %s",
                corr_prot.threshold_multiplier,
                {k.value: f"{v:.1%}" for k, v in self.thresholds.items()},
            )

    # ------------------------------------------------------------------
    # Protection effectiveness
    # ------------------------------------------------------------------

    @staticmethod
    def measure_effectiveness(
        protected_equity: pd.Series,
        unprotected_equity: pd.Series,
        events: List[ProtectionEvent],
    ) -> ProtectionEffectiveness:
        """Compare max drawdown with vs without protection."""
        def _max_dd(eq: pd.Series) -> float:
            hwm = eq.expanding().max()
            dd = 1.0 - eq / hwm
            return float(dd.max()) if not dd.empty else 0.0

        mdd_w = _max_dd(protected_equity)
        mdd_wo = _max_dd(unprotected_equity)
        reduction = (mdd_wo - mdd_w) / mdd_wo if mdd_wo > 1e-12 else 0.0
        n_int = sum(1 for e in events if e.new_level != ProtectionLevel.GREEN)

        # Average recovery: days between escalation and return to green
        recovery_days: List[int] = []
        esc_idx = None
        for i, e in enumerate(events):
            if e.new_level != ProtectionLevel.GREEN and esc_idx is None:
                esc_idx = i
            elif e.new_level == ProtectionLevel.GREEN and esc_idx is not None:
                recovery_days.append(i - esc_idx)
                esc_idx = None
        avg_rec = float(np.mean(recovery_days)) if recovery_days else 0.0

        return ProtectionEffectiveness(
            max_drawdown_with=mdd_w,
            max_drawdown_without=mdd_wo,
            reduction_pct=reduction,
            n_interventions=n_int,
            avg_recovery_days=avg_rec,
        )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> ProtectionLevel:
        return self._current_level

    @property
    def high_water_mark(self) -> float:
        return self._hwm

    @property
    def events(self) -> List[ProtectionEvent]:
        return list(self._events)

    @property
    def state_history(self) -> List[DrawdownState]:
        return list(self._state_history)

    def reset(self) -> None:
        self._hwm = 0.0
        self._current_level = ProtectionLevel.GREEN
        self._events.clear()
        self._state_history.clear()
        self._strategy_hwms.clear()

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 720, height: int = 210, color: str = "#2980b9",
        thresholds: Optional[Dict[str, Tuple[float, str]]] = None,
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        vmin = min(values)
        vmax = max(values)
        extra_lines = thresholds or {}
        for _, (tv, _) in extra_lines.items():
            vmax = max(vmax, tv)
        if vmax <= vmin:
            vmax = vmin + 0.01
        pad_l, pad_r, pad_t, pad_b = 55, 20, 28, 25
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
        for label, (tv, tc) in extra_lines.items():
            yy = ty(tv)
            p.append(f'<line x1="{pad_l}" y1="{yy:.0f}" x2="{width - pad_r}" '
                     f'y2="{yy:.0f}" stroke="{tc}" stroke-width="1" stroke-dasharray="4,3"/>')
            p.append(f'<text x="{width - pad_r + 2}" y="{yy + 3:.0f}" font-size="8" '
                     f'fill="{tc}">{label}</text>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                      for i, v in enumerate(values))
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_timeline(
        events: List[ProtectionEvent],
        total_days: int,
        width: int = 720, height: int = 50,
    ) -> str:
        if not events or total_days < 1:
            return ""
        colors = {
            ProtectionLevel.GREEN: "#27ae60", ProtectionLevel.YELLOW: "#f1c40f",
            ProtectionLevel.ORANGE: "#e67e22", ProtectionLevel.RED: "#e74c3c",
        }
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        # background green
        p.append(f'<rect x="0" y="0" width="{width}" height="{height - 16}" fill="#27ae60"/>')
        # overlay events
        for ev in events:
            c = colors.get(ev.new_level, "#999")
            # approximate x position
            idx = min(len(events) - 1, events.index(ev))
            x = idx / max(len(events), 1) * width
            p.append(f'<rect x="{x:.0f}" y="0" width="{max(width / total_days * 5, 3):.0f}" '
                     f'height="{height - 16}" fill="{c}"/>')
        # legend
        lx = 5
        for lvl, c in colors.items():
            p.append(f'<rect x="{lx}" y="{height - 12}" width="8" height="8" fill="{c}"/>')
            p.append(f'<text x="{lx + 11}" y="{height - 4}" font-size="8" fill="#333">{lvl.value}</text>')
            lx += 65
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        equity_series: Optional[pd.Series] = None,
        effectiveness: Optional[ProtectionEffectiveness] = None,
        recovery: Optional[RecoveryEstimate] = None,
        output_path: str = "reports/drawdown_protection.html",
    ) -> str:
        """HTML report: drawdown path, response timeline, effectiveness."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Drawdown chart
        dd_vals = [s.drawdown for s in self._state_history]
        th_lines = {
            "yellow": (self.thresholds.get(ProtectionLevel.YELLOW, 0.03), "#f1c40f"),
            "orange": (self.thresholds.get(ProtectionLevel.ORANGE, 0.05), "#e67e22"),
            "red": (self.thresholds.get(ProtectionLevel.RED, 0.08), "#e74c3c"),
        }
        dd_svg = self._svg_line(dd_vals, "Drawdown Path", color="#e74c3c",
                                 thresholds=th_lines)

        # Equity chart
        eq_vals = [s.equity for s in self._state_history]
        eq_svg = self._svg_line(eq_vals, "Equity Curve", color="#2980b9")

        # Timeline
        timeline_svg = self._svg_timeline(
            self._events, max(len(self._state_history), 1))

        # Event table
        ev_rows = []
        for e in self._events:
            ds = e.date.strftime("%Y-%m-%d") if hasattr(e.date, "strftime") else str(e.date)
            ev_rows.append(
                f"<tr><td>{ds}</td><td class='{e.old_level.value}'>{e.old_level.value}</td>"
                f"<td class='{e.new_level.value}'>{e.new_level.value}</td>"
                f"<td>{e.drawdown:.2%}</td><td>{e.action}</td>"
                f"<td>{e.trigger}</td></tr>")

        # Effectiveness
        eff_html = ""
        if effectiveness is not None:
            e = effectiveness
            eff_html = f"""
<h2>Protection Effectiveness</h2>
<table class="m"><tr><th>Max DD (Protected)</th><th>Max DD (Unprotected)</th>
<th>Reduction</th><th>Interventions</th><th>Avg Recovery Days</th></tr>
<tr><td>{e.max_drawdown_with:.2%}</td><td>{e.max_drawdown_without:.2%}</td>
<td>{e.reduction_pct:.1%}</td><td>{e.n_interventions}</td>
<td>{e.avg_recovery_days:.1f}</td></tr></table>"""

        # Recovery
        rec_html = ""
        if recovery is not None:
            r = recovery
            rec_html = f"""
<h2>Recovery Estimate</h2>
<table class="m"><tr><th>Current DD</th><th>Expected Days</th>
<th>80th Pctile</th><th>Historical Avg</th><th>Episodes</th></tr>
<tr><td>{r.current_drawdown:.2%}</td><td>{r.expected_days:.0f}</td>
<td>{r.confidence_80:.0f}</td><td>{r.historical_avg_recovery:.0f}</td>
<td>{r.n_historical_episodes}</td></tr></table>"""

        # Current state
        cur = self._state_history[-1] if self._state_history else None
        cur_html = ""
        if cur:
            level_colors = {"green": "#27ae60", "yellow": "#f1c40f",
                            "orange": "#e67e22", "red": "#e74c3c"}
            lc = level_colors.get(cur.level.value, "#999")
            cur_html = f"""
<p><strong>Current Level:</strong>
<span style="background:{lc};color:#fff;padding:3px 10px;border-radius:8px;
font-weight:bold">{cur.level.value.upper()}</span>
— DD {cur.drawdown:.2%} — Action: {cur.action} — Size mult: {cur.size_multiplier:.0%}</p>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Drawdown Protection</title>
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
td.green {{ color: #27ae60; font-weight: bold; }}
td.yellow {{ color: #f1c40f; font-weight: bold; }}
td.orange {{ color: #e67e22; font-weight: bold; }}
td.red {{ color: #e74c3c; font-weight: bold; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Drawdown Protection Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Thresholds:</strong>
   Yellow &ge; {self.thresholds.get(ProtectionLevel.YELLOW, 0):.1%} |
   Orange &ge; {self.thresholds.get(ProtectionLevel.ORANGE, 0):.1%} |
   Red &ge; {self.thresholds.get(ProtectionLevel.RED, 0):.1%}</p>
{cur_html}
</div>

<h2>Drawdown Path</h2>
{dd_svg}

<h2>Equity Curve</h2>
{eq_svg}

<h2>Response Timeline</h2>
{timeline_svg}

<h2>Protection Events</h2>
<table><tr><th>Date</th><th>From</th><th>To</th><th>Drawdown</th>
<th>Action</th><th>Trigger</th></tr>
{''.join(ev_rows)}</table>

{eff_html}
{rec_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Drawdown protection report -> %s", path)
        return str(path)
