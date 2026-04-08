"""Real-time crisis hedge monitoring dashboard — tracks VIX tiers, scale
adjustments, hedge cost vs budget, drawdown controller state, and recovery
detection during paper trading.

Designed to be called periodically (every tick or every N minutes) and
to produce daily/weekly summary HTML reports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── VIX tier labels ─────────────────────────────────────────────────────────
TIER_NORMAL = "NORMAL"          # VIX < 25
TIER_ELEVATED = "ELEVATED"      # 25 <= VIX < 35
TIER_HIGH = "HIGH"              # 35 <= VIX < 50
TIER_EXTREME = "EXTREME"        # VIX >= 50

ANNUAL_HEDGE_BUDGET_PCT = 0.33  # 0.33% annual drag budget (from EXP-880)
TRADING_DAYS_PER_YEAR = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ScaleAdjustment:
    """Single logged scale change."""
    timestamp: str
    vix: float
    prev_scale: float
    new_scale: float
    tier: str
    reason: str
    current_dd: float = 0.0
    regime: str = ""


@dataclass
class HedgeCostEntry:
    """Daily hedge cost record."""
    date: str
    put_cost: float            # dollars spent on put overlay
    opportunity_cost: float    # lost return from delevering
    total_cost: float
    portfolio_value: float
    cost_pct: float            # as fraction of portfolio


@dataclass
class DDControllerState:
    """Drawdown controller snapshot."""
    current_dd: float
    dd_start_threshold: float
    dd_full_threshold: float
    dd_scale: float
    is_active: bool           # True if DD > dd_start


@dataclass
class RecoveryState:
    """Recovery detection snapshot."""
    is_recovering: bool
    momentum_confirmed: bool
    vix_normalised: bool
    progress: float           # 0-1 ramp
    days_in_recovery: int
    estimated_full_recovery_date: str = ""


@dataclass
class MonitorState:
    """Complete monitor snapshot at one point in time."""
    timestamp: str
    vix: float
    tier: str
    scale_factor: float
    dd_controller: DDControllerState
    recovery: RecoveryState
    put_overlay_active: bool
    put_cost_today: float = 0.0
    regime: str = ""


@dataclass
class DailySummary:
    """Daily hedge monitoring summary."""
    date: str
    avg_vix: float
    min_vix: float
    max_vix: float
    avg_scale: float
    min_scale: float
    n_adjustments: int
    tier_distribution: Dict[str, int]  # tier → count of ticks in that tier
    total_hedge_cost: float
    cost_vs_budget_pct: float          # actual / budgeted
    dd_peak: float
    recovery_active: bool
    n_ticks: int


@dataclass
class WeeklySummary:
    """Weekly aggregate from daily summaries."""
    week_start: str
    week_end: str
    avg_vix: float
    avg_scale: float
    total_hedge_cost: float
    cumulative_cost_annualised_pct: float
    n_adjustments: int
    worst_dd: float
    days_in_elevated_plus: int   # days where tier >= ELEVATED
    daily_summaries: List[DailySummary] = field(default_factory=list)


# ── Core Monitor ────────────────────────────────────────────────────────────
class CrisisHedgeMonitor:
    """Tracks and reports on crisis hedge activity in real-time."""

    def __init__(
        self,
        vix_reduce: float = 25.0,
        vix_minimum: float = 35.0,
        vix_extreme: float = 50.0,
        dd_start: float = 0.05,
        dd_full: float = 0.12,
        annual_budget_pct: float = ANNUAL_HEDGE_BUDGET_PCT,
        starting_capital: float = 100_000.0,
    ) -> None:
        self.vix_reduce = vix_reduce
        self.vix_minimum = vix_minimum
        self.vix_extreme = vix_extreme
        self.dd_start = dd_start
        self.dd_full = dd_full
        self.annual_budget = annual_budget_pct
        self.starting_capital = starting_capital

        self._adjustments: List[ScaleAdjustment] = []
        self._cost_entries: List[HedgeCostEntry] = []
        self._states: List[MonitorState] = []
        self._prev_scale: float = 1.0
        self._cumulative_cost: float = 0.0
        self._current_date: str = ""
        self._daily_ticks: int = 0
        self._daily_cost: float = 0.0
        self._daily_vix: List[float] = []

    # ── Tick processing ─────────────────────────────────────────────────────
    def record_tick(
        self,
        vix: float,
        scale_factor: float,
        current_dd: float = 0.0,
        regime: str = "bull",
        put_overlay_active: bool = False,
        put_cost: float = 0.0,
        momentum_confirmed: bool = False,
        vix_normalised: bool = False,
        recovery_progress: float = 0.0,
        recovery_days: int = 0,
        portfolio_value: float = 0.0,
    ) -> MonitorState:
        """Record a single monitoring tick."""
        now = _now()
        today = now[:10]

        # Classify VIX tier
        tier = self._classify_tier(vix)

        # Log scale adjustment if changed
        if abs(scale_factor - self._prev_scale) > 0.005:
            reason = self._build_reason(vix, current_dd, regime, tier)
            self._adjustments.append(ScaleAdjustment(
                timestamp=now, vix=vix,
                prev_scale=round(self._prev_scale, 4),
                new_scale=round(scale_factor, 4),
                tier=tier, reason=reason,
                current_dd=current_dd, regime=regime,
            ))

        self._prev_scale = scale_factor

        # Track hedge cost
        self._cumulative_cost += put_cost
        self._daily_cost += put_cost
        self._daily_vix.append(vix)
        self._daily_ticks += 1

        # DD controller state
        dd_state = DDControllerState(
            current_dd=current_dd,
            dd_start_threshold=self.dd_start,
            dd_full_threshold=self.dd_full,
            dd_scale=self._dd_to_scale(current_dd),
            is_active=current_dd > self.dd_start,
        )

        # Recovery state
        rec_state = RecoveryState(
            is_recovering=momentum_confirmed and vix_normalised,
            momentum_confirmed=momentum_confirmed,
            vix_normalised=vix_normalised,
            progress=recovery_progress,
            days_in_recovery=recovery_days,
        )

        state = MonitorState(
            timestamp=now, vix=vix, tier=tier,
            scale_factor=scale_factor,
            dd_controller=dd_state,
            recovery=rec_state,
            put_overlay_active=put_overlay_active,
            put_cost_today=put_cost,
            regime=regime,
        )
        self._states.append(state)

        # Date rollover for daily tracking
        if today != self._current_date and self._current_date:
            self._flush_daily(self._current_date, portfolio_value)
        self._current_date = today

        return state

    # ── Daily/weekly summaries ──────────────────────────────────────────────
    def get_daily_summary(self, date_str: Optional[str] = None) -> Optional[DailySummary]:
        """Get summary for a specific date (or latest)."""
        target = date_str or self._current_date
        day_states = [s for s in self._states if s.timestamp[:10] == target]
        if not day_states:
            return None

        vix_vals = [s.vix for s in day_states]
        scales = [s.scale_factor for s in day_states]
        day_adjs = [a for a in self._adjustments if a.timestamp[:10] == target]
        tiers = {}
        for s in day_states:
            tiers[s.tier] = tiers.get(s.tier, 0) + 1

        day_cost = sum(s.put_cost_today for s in day_states)
        budget_daily = self.annual_budget / 100 * self.starting_capital / TRADING_DAYS_PER_YEAR
        cost_vs_budget = day_cost / budget_daily * 100 if budget_daily > 0 else 0

        dd_peak = max(s.dd_controller.current_dd for s in day_states)
        recovery = any(s.recovery.is_recovering for s in day_states)

        return DailySummary(
            date=target,
            avg_vix=round(float(np.mean(vix_vals)), 2),
            min_vix=round(float(min(vix_vals)), 2),
            max_vix=round(float(max(vix_vals)), 2),
            avg_scale=round(float(np.mean(scales)), 4),
            min_scale=round(float(min(scales)), 4),
            n_adjustments=len(day_adjs),
            tier_distribution=tiers,
            total_hedge_cost=round(day_cost, 2),
            cost_vs_budget_pct=round(cost_vs_budget, 1),
            dd_peak=round(dd_peak, 4),
            recovery_active=recovery,
            n_ticks=len(day_states),
        )

    def get_weekly_summary(self, n_days: int = 5) -> WeeklySummary:
        """Get summary for the last N trading days."""
        dates = sorted(set(s.timestamp[:10] for s in self._states))
        recent = dates[-n_days:] if len(dates) >= n_days else dates

        dailies = []
        for d in recent:
            ds = self.get_daily_summary(d)
            if ds:
                dailies.append(ds)

        if not dailies:
            return WeeklySummary("", "", 0, 1.0, 0, 0, 0, 0, 0)

        total_cost = sum(d.total_hedge_cost for d in dailies)
        # Annualise: cost_per_day * 252 / starting_capital * 100
        days = len(dailies)
        ann_cost_pct = (total_cost / days * TRADING_DAYS_PER_YEAR / self.starting_capital * 100) if days > 0 else 0

        elevated_days = sum(
            1 for d in dailies
            if any(t in d.tier_distribution for t in (TIER_ELEVATED, TIER_HIGH, TIER_EXTREME))
        )

        return WeeklySummary(
            week_start=dailies[0].date,
            week_end=dailies[-1].date,
            avg_vix=round(float(np.mean([d.avg_vix for d in dailies])), 2),
            avg_scale=round(float(np.mean([d.avg_scale for d in dailies])), 4),
            total_hedge_cost=round(total_cost, 2),
            cumulative_cost_annualised_pct=round(ann_cost_pct, 3),
            n_adjustments=sum(d.n_adjustments for d in dailies),
            worst_dd=round(max(d.dd_peak for d in dailies), 4),
            days_in_elevated_plus=elevated_days,
            daily_summaries=dailies,
        )

    # ── Accessors ───────────────────────────────────────────────────────────
    @property
    def adjustments(self) -> List[ScaleAdjustment]:
        return list(self._adjustments)

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def total_ticks(self) -> int:
        return len(self._states)

    def cost_vs_budget(self, days_elapsed: int) -> Dict[str, float]:
        """Compare realised hedge cost against annual budget."""
        if days_elapsed <= 0:
            return {"budget": 0, "actual": 0, "utilisation_pct": 0}
        budget_so_far = self.annual_budget / 100 * self.starting_capital * days_elapsed / TRADING_DAYS_PER_YEAR
        actual = self._cumulative_cost
        util = actual / budget_so_far * 100 if budget_so_far > 0 else 0
        return {
            "budget": round(budget_so_far, 2),
            "actual": round(actual, 2),
            "utilisation_pct": round(util, 1),
        }

    def reset(self) -> None:
        self._adjustments.clear()
        self._cost_entries.clear()
        self._states.clear()
        self._prev_scale = 1.0
        self._cumulative_cost = 0.0
        self._daily_ticks = 0
        self._daily_cost = 0.0
        self._daily_vix.clear()

    # ── Internal ────────────────────────────────────────────────────────────
    def _classify_tier(self, vix: float) -> str:
        if vix >= self.vix_extreme:
            return TIER_EXTREME
        if vix >= self.vix_minimum:
            return TIER_HIGH
        if vix >= self.vix_reduce:
            return TIER_ELEVATED
        return TIER_NORMAL

    def _dd_to_scale(self, dd: float) -> float:
        if dd <= self.dd_start:
            return 1.0
        if dd >= self.dd_full:
            return 0.20
        t = (dd - self.dd_start) / (self.dd_full - self.dd_start)
        return 1.0 - t * 0.60

    def _build_reason(self, vix: float, dd: float, regime: str, tier: str) -> str:
        parts = [f"VIX={vix:.1f}({tier})"]
        if dd > self.dd_start:
            parts.append(f"DD={dd:.1%}")
        if regime in ("crash", "high_vol"):
            parts.append(f"regime={regime}")
        return "; ".join(parts)

    def _flush_daily(self, date_str: str, portfolio_value: float) -> None:
        if self._daily_ticks == 0:
            return
        cost_pct = self._daily_cost / max(portfolio_value, self.starting_capital) * 100
        self._cost_entries.append(HedgeCostEntry(
            date=date_str,
            put_cost=round(self._daily_cost, 2),
            opportunity_cost=0.0,
            total_cost=round(self._daily_cost, 2),
            portfolio_value=portfolio_value or self.starting_capital,
            cost_pct=round(cost_pct, 4),
        ))
        self._daily_ticks = 0
        self._daily_cost = 0.0
        self._daily_vix.clear()

    # ── HTML Report ─────────────────────────────────────────────────────────
    def generate_report(
        self,
        output_path: str = "reports/crisis_hedge_monitor.html",
        period: str = "daily",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if period == "weekly":
            summary = self.get_weekly_summary()
            title = f"Weekly Hedge Report ({summary.week_start} — {summary.week_end})"
        else:
            summary = self.get_daily_summary()
            title = f"Daily Hedge Report ({summary.date if summary else 'N/A'})"

        latest = self._states[-1] if self._states else None
        cards = self._html_cards(latest, summary)
        adj_tbl = self._html_adjustments()
        cost_sec = self._html_cost_analysis()
        tier_sec = self._html_tier_distribution(summary)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#fff;color:#1e293b;padding:24px;max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1rem;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin:20px 0 10px}}
.sub{{color:#64748b;font-size:.85rem;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px}}
.card .lbl{{font-size:.7rem;color:#64748b;text-transform:uppercase}}
.card .val{{font-size:1.2rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem;margin-bottom:16px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #e2e8f0}}
th{{color:#64748b;background:#f8fafc}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}.warn{{color:#d97706}}
.tier{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600}}
.tier.NORMAL{{background:#dcfce7;color:#166534}}
.tier.ELEVATED{{background:#fef9c3;color:#854d0e}}
.tier.HIGH{{background:#fed7aa;color:#9a3412}}
.tier.EXTREME{{background:#fecaca;color:#991b1b}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">Crisis Hedge V2 Monitor &middot; Budget: {self.annual_budget}%/yr</p>
{cards}
{tier_sec}
{adj_tbl}
{cost_sec}
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        return path

    def _html_cards(self, latest: Optional[MonitorState], summary: Any) -> str:
        if not latest:
            return "<p>No data recorded yet.</p>"
        dd = latest.dd_controller
        rec = latest.recovery
        cost = self.cost_vs_budget(len(set(s.timestamp[:10] for s in self._states)))
        return f"""<div class="grid">
<div class="card"><div class="lbl">VIX</div><div class="val">{latest.vix:.1f}</div></div>
<div class="card"><div class="lbl">Tier</div><div class="val"><span class="tier {latest.tier}">{latest.tier}</span></div></div>
<div class="card"><div class="lbl">Scale</div><div class="val {'warn' if latest.scale_factor<0.8 else ''}">{latest.scale_factor:.0%}</div></div>
<div class="card"><div class="lbl">Drawdown</div><div class="val {'neg' if dd.current_dd>0.05 else ''}">{dd.current_dd:.1%}</div></div>
<div class="card"><div class="lbl">DD Scale</div><div class="val">{dd.dd_scale:.0%}</div></div>
<div class="card"><div class="lbl">DD Active</div><div class="val {'neg' if dd.is_active else 'pos'}">{'YES' if dd.is_active else 'No'}</div></div>
<div class="card"><div class="lbl">Recovering</div><div class="val {'pos' if rec.is_recovering else ''}">{rec.progress:.0%}</div></div>
<div class="card"><div class="lbl">Put Overlay</div><div class="val">{'ON' if latest.put_overlay_active else 'Off'}</div></div>
<div class="card"><div class="lbl">Hedge Cost</div><div class="val">${self._cumulative_cost:,.0f}</div></div>
<div class="card"><div class="lbl">Budget Used</div><div class="val {'warn' if cost['utilisation_pct']>80 else ''}">{cost['utilisation_pct']:.0f}%</div></div>
<div class="card"><div class="lbl">Adjustments</div><div class="val">{len(self._adjustments)}</div></div>
<div class="card"><div class="lbl">Regime</div><div class="val">{latest.regime}</div></div>
</div>"""

    def _html_adjustments(self) -> str:
        if not self._adjustments:
            return "<h2>Scale Adjustments</h2><p>No adjustments recorded.</p>"
        rows = ""
        for a in self._adjustments[-30:]:
            direction = "↓" if a.new_scale < a.prev_scale else "↑"
            cls = "neg" if a.new_scale < a.prev_scale else "pos"
            rows += (f"<tr><td>{a.timestamp}</td><td>{a.vix:.1f}</td>"
                     f"<td><span class='tier {a.tier}'>{a.tier}</span></td>"
                     f"<td>{a.prev_scale:.0%}</td>"
                     f'<td class="{cls}">{direction} {a.new_scale:.0%}</td>'
                     f"<td>{a.reason}</td></tr>")
        return f"""<h2>Scale Adjustments (last 30)</h2>
<table><thead><tr><th>Time</th><th>VIX</th><th>Tier</th><th>From</th><th>To</th><th>Reason</th></tr></thead>
<tbody>{rows}</tbody></table>"""

    def _html_cost_analysis(self) -> str:
        if not self._cost_entries:
            return ""
        rows = ""
        for c in self._cost_entries[-20:]:
            rows += (f"<tr><td>{c.date}</td><td>${c.put_cost:.2f}</td>"
                     f"<td>${c.total_cost:.2f}</td><td>{c.cost_pct:.3f}%</td></tr>")
        return f"""<h2>Hedge Cost History</h2>
<table><thead><tr><th>Date</th><th>Put Cost</th><th>Total</th><th>% of Portfolio</th></tr></thead>
<tbody>{rows}</tbody></table>"""

    def _html_tier_distribution(self, summary: Any) -> str:
        if summary is None or not hasattr(summary, "tier_distribution"):
            return ""
        dist = getattr(summary, "tier_distribution", {})
        if not dist:
            return ""
        total = sum(dist.values())
        rows = ""
        for tier in [TIER_NORMAL, TIER_ELEVATED, TIER_HIGH, TIER_EXTREME]:
            count = dist.get(tier, 0)
            pct = count / total * 100 if total > 0 else 0
            rows += f'<tr><td><span class="tier {tier}">{tier}</span></td><td>{count}</td><td>{pct:.0f}%</td></tr>'
        return f"""<h2>VIX Tier Distribution</h2>
<table><thead><tr><th>Tier</th><th>Ticks</th><th>%</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
