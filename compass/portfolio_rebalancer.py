"""
Portfolio rebalancer — computes optimal rebalance trades and monitors drift.

Rebalance triggers:
  - Calendar:    weekly / custom cadence
  - Threshold:   when any position drifts beyond tolerance (default 5%)
  - Regime:      immediate rebalance on vol-regime change

Tax-aware rebalancing prioritises selling losers first (harvest losses).
Transaction cost minimisation skips small trades below a cost threshold.

Integrates with portfolio_optimizer.py target weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class TriggerType(str, Enum):
    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    REGIME = "regime"
    MANUAL = "manual"


DEFAULT_DRIFT_THRESHOLD = 0.05   # 5%
DEFAULT_CALENDAR_DAYS = 7        # weekly
DEFAULT_MIN_TRADE_SIZE = 0.005   # skip trades < 0.5% of portfolio


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Current portfolio position."""
    name: str
    current_weight: float
    target_weight: float
    unrealised_pnl: float = 0.0   # positive = gain, negative = loss
    cost_basis: float = 0.0


@dataclass
class RebalanceTrade:
    """A single trade needed to rebalance."""
    name: str
    current_weight: float
    target_weight: float
    trade_weight: float          # positive = buy, negative = sell
    is_tax_harvest: bool = False
    estimated_cost: float = 0.0


@dataclass
class DriftSnapshot:
    """Point-in-time drift measurement."""
    date: datetime
    drifts: Dict[str, float]     # {name: abs(current - target)}
    max_drift: float
    max_drift_name: str
    threshold_breached: bool


@dataclass
class RebalanceEvent:
    """Full rebalance record."""
    date: datetime
    trigger: TriggerType
    trades: List[RebalanceTrade]
    total_turnover: float        # sum of |trade_weight|
    estimated_cost: float
    tax_harvested: float         # total losses harvested
    drift_before: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Alert when a position drifts beyond threshold."""
    date: datetime
    name: str
    drift: float
    threshold: float


# ---------------------------------------------------------------------------
# Core rebalancer
# ---------------------------------------------------------------------------

class PortfolioRebalancer:
    """Computes rebalance trades, monitors drift, manages triggers.

    Args:
        drift_threshold: Fractional threshold triggering rebalance (0.05 = 5%).
        calendar_days: Days between calendar-triggered rebalances.
        min_trade_size: Minimum trade size (fraction) — trades below are skipped.
        cost_per_unit: Transaction cost per unit of turnover.
        tax_aware: Enable tax-loss harvesting prioritisation.
    """

    def __init__(
        self,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        calendar_days: int = DEFAULT_CALENDAR_DAYS,
        min_trade_size: float = DEFAULT_MIN_TRADE_SIZE,
        cost_per_unit: float = 0.001,
        tax_aware: bool = True,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.calendar_days = calendar_days
        self.min_trade_size = min_trade_size
        self.cost_per_unit = cost_per_unit
        self.tax_aware = tax_aware

        self._last_rebalance: Optional[datetime] = None
        self._events: List[RebalanceEvent] = []
        self._drift_history: List[DriftSnapshot] = []
        self._alerts: List[DriftAlert] = []

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    def compute_drift(
        self, positions: List[Position], date: Optional[datetime] = None,
    ) -> DriftSnapshot:
        """Measure current-vs-target drift for every position."""
        dt = date or datetime.now()
        drifts: Dict[str, float] = {}
        for p in positions:
            drifts[p.name] = abs(p.current_weight - p.target_weight)

        max_d = max(drifts.values()) if drifts else 0.0
        max_name = max(drifts, key=drifts.get) if drifts else ""

        snap = DriftSnapshot(
            date=dt, drifts=drifts, max_drift=max_d,
            max_drift_name=max_name,
            threshold_breached=max_d >= self.drift_threshold,
        )
        self._drift_history.append(snap)

        # Raise alerts for any position over threshold
        for name, d in drifts.items():
            if d >= self.drift_threshold:
                self._alerts.append(DriftAlert(
                    date=dt, name=name, drift=d,
                    threshold=self.drift_threshold,
                ))

        return snap

    # ------------------------------------------------------------------
    # Trigger checks
    # ------------------------------------------------------------------

    def should_rebalance_calendar(self, now: Optional[datetime] = None) -> bool:
        """True if enough time has passed since last rebalance."""
        now = now or datetime.now()
        if self._last_rebalance is None:
            return True
        return (now - self._last_rebalance) >= timedelta(days=self.calendar_days)

    def should_rebalance_threshold(self, positions: List[Position]) -> bool:
        """True if any position exceeds drift threshold."""
        for p in positions:
            if abs(p.current_weight - p.target_weight) >= self.drift_threshold:
                return True
        return False

    def check_triggers(
        self,
        positions: List[Position],
        now: Optional[datetime] = None,
        regime_changed: bool = False,
    ) -> Optional[TriggerType]:
        """Return the highest-priority trigger that fires, or None."""
        if regime_changed:
            return TriggerType.REGIME
        if self.should_rebalance_threshold(positions):
            return TriggerType.THRESHOLD
        if self.should_rebalance_calendar(now):
            return TriggerType.CALENDAR
        return None

    # ------------------------------------------------------------------
    # Trade computation
    # ------------------------------------------------------------------

    def compute_trades(
        self, positions: List[Position],
    ) -> List[RebalanceTrade]:
        """Compute trades to move from current to target weights.

        Tax-aware mode sorts sells so losses are harvested first.
        Trades below min_trade_size are filtered out.
        """
        raw: List[RebalanceTrade] = []
        for p in positions:
            delta = p.target_weight - p.current_weight
            if abs(delta) < self.min_trade_size:
                continue
            is_harvest = self.tax_aware and delta < 0 and p.unrealised_pnl < 0
            cost = abs(delta) * self.cost_per_unit
            raw.append(RebalanceTrade(
                name=p.name,
                current_weight=p.current_weight,
                target_weight=p.target_weight,
                trade_weight=delta,
                is_tax_harvest=is_harvest,
                estimated_cost=cost,
            ))

        if self.tax_aware:
            # Sort: tax-harvest sells first, then other sells, then buys
            raw.sort(key=lambda t: (
                0 if t.is_tax_harvest else (1 if t.trade_weight < 0 else 2),
                -abs(t.trade_weight),
            ))

        return raw

    def compute_trades_with_cost_limit(
        self,
        positions: List[Position],
        max_cost: float,
    ) -> List[RebalanceTrade]:
        """Compute trades but stop adding once estimated cost hits max_cost."""
        all_trades = self.compute_trades(positions)
        kept: List[RebalanceTrade] = []
        running_cost = 0.0
        for t in all_trades:
            if running_cost + t.estimated_cost > max_cost:
                continue
            kept.append(t)
            running_cost += t.estimated_cost
        return kept

    # ------------------------------------------------------------------
    # Execute rebalance
    # ------------------------------------------------------------------

    def rebalance(
        self,
        positions: List[Position],
        trigger: TriggerType = TriggerType.MANUAL,
        date: Optional[datetime] = None,
    ) -> RebalanceEvent:
        """Compute trades and record the rebalance event."""
        dt = date or datetime.now()
        drift_snap = self.compute_drift(positions, date=dt)
        trades = self.compute_trades(positions)

        total_turnover = sum(abs(t.trade_weight) for t in trades)
        est_cost = sum(t.estimated_cost for t in trades)
        harvested = sum(
            abs(p.unrealised_pnl)
            for p in positions
            for t in trades
            if t.name == p.name and t.is_tax_harvest
        )

        event = RebalanceEvent(
            date=dt, trigger=trigger, trades=trades,
            total_turnover=total_turnover,
            estimated_cost=est_cost,
            tax_harvested=harvested,
            drift_before=drift_snap.drifts,
        )
        self._events.append(event)
        self._last_rebalance = dt
        return event

    # ------------------------------------------------------------------
    # Integration with portfolio_optimizer target weights
    # ------------------------------------------------------------------

    @staticmethod
    def positions_from_optimizer(
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        unrealised_pnl: Optional[Dict[str, float]] = None,
    ) -> List[Position]:
        """Build Position list from optimizer output dicts."""
        pnl = unrealised_pnl or {}
        all_names = set(current_weights) | set(target_weights)
        return [
            Position(
                name=n,
                current_weight=current_weights.get(n, 0.0),
                target_weight=target_weights.get(n, 0.0),
                unrealised_pnl=pnl.get(n, 0.0),
            )
            for n in sorted(all_names)
        ]

    # ------------------------------------------------------------------
    # History / alerts
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[RebalanceEvent]:
        return list(self._events)

    @property
    def drift_history(self) -> List[DriftSnapshot]:
        return list(self._drift_history)

    @property
    def alerts(self) -> List[DriftAlert]:
        return list(self._alerts)

    @property
    def last_rebalance(self) -> Optional[datetime]:
        return self._last_rebalance

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_drift_chart(
        history: List[DriftSnapshot],
        threshold: float,
        width: int = 700, height: int = 220,
    ) -> str:
        """SVG line chart of max drift over time with threshold line."""
        if len(history) < 2:
            return ""
        vals = [s.max_drift for s in history]
        n = len(vals)
        y_max = max(max(vals) * 1.2, threshold * 1.5)
        pad_l, pad_r, pad_t, pad_b = 50, 15, 25, 30
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

        def tx(i: int) -> float:
            return pad_l + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return pad_t + (1 - v / y_max) * ph

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        parts.append(
            f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
            f'font-weight="bold" fill="#1a1a2e">Max Portfolio Drift</text>'
        )
        # threshold
        thy = ty(threshold)
        parts.append(
            f'<line x1="{pad_l}" y1="{thy:.0f}" x2="{width - pad_r}" '
            f'y2="{thy:.0f}" stroke="#e74c3c" stroke-width="1" stroke-dasharray="4,3"/>'
        )
        parts.append(
            f'<text x="{width - pad_r + 2}" y="{thy + 4:.0f}" font-size="9" '
            f'fill="#e74c3c">{threshold:.0%}</text>'
        )
        # line
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
            for i, v in enumerate(vals)
        )
        parts.append(f'<path d="{d}" fill="none" stroke="#2980b9" stroke-width="2"/>')
        # breach dots
        for i, s in enumerate(history):
            if s.threshold_breached:
                parts.append(
                    f'<circle cx="{tx(i):.1f}" cy="{ty(vals[i]):.1f}" r="4" '
                    f'fill="#e74c3c"/>'
                )
        parts.append("</svg>")
        return "\n".join(parts)

    def generate_report(
        self,
        positions: List[Position],
        output_path: str = "reports/rebalance.html",
    ) -> str:
        """HTML report: drift chart, rebalance history, cost analysis."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        drift_svg = self._svg_drift_chart(self._drift_history, self.drift_threshold)

        # --- Event table ---
        event_rows = []
        for e in self._events:
            ds = e.date.strftime("%Y-%m-%d %H:%M") if hasattr(e.date, "strftime") else str(e.date)
            event_rows.append(
                f"<tr><td>{ds}</td><td>{e.trigger.value}</td>"
                f"<td>{len(e.trades)}</td><td>{e.total_turnover:.2%}</td>"
                f"<td>{e.estimated_cost:.4f}</td>"
                f"<td>{e.tax_harvested:.4f}</td></tr>"
            )

        # --- Current drift table ---
        drift_rows = []
        for p in positions:
            d = abs(p.current_weight - p.target_weight)
            cls = ' class="breach"' if d >= self.drift_threshold else ""
            drift_rows.append(
                f"<tr><td>{p.name}</td><td>{p.current_weight:.2%}</td>"
                f"<td>{p.target_weight:.2%}</td>"
                f"<td{cls}>{d:.2%}</td>"
                f"<td>{p.unrealised_pnl:+.4f}</td></tr>"
            )

        # --- Cost summary ---
        total_cost = sum(e.estimated_cost for e in self._events)
        total_harvest = sum(e.tax_harvested for e in self._events)

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Portfolio Rebalancer Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
td.breach {{ color: #e74c3c; font-weight: bold; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Portfolio Rebalancer Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Drift Threshold:</strong> {self.drift_threshold:.0%} |
   <strong>Calendar:</strong> every {self.calendar_days}d |
   <strong>Tax-Aware:</strong> {'Yes' if self.tax_aware else 'No'}</p>
<p><strong>Total Rebalances:</strong> {len(self._events)} |
   <strong>Total Cost:</strong> {total_cost:.4f} |
   <strong>Tax Harvested:</strong> {total_harvest:.4f}</p>
</div>

<h2>Drift Over Time</h2>
{drift_svg}

<h2>Current Positions &amp; Drift</h2>
<table><tr><th>Name</th><th>Current</th><th>Target</th><th>Drift</th><th>Unreal P&amp;L</th></tr>
{''.join(drift_rows)}</table>

<h2>Rebalance History</h2>
<table><tr><th>Date</th><th>Trigger</th><th>Trades</th><th>Turnover</th>
<th>Cost</th><th>Tax Harvested</th></tr>
{''.join(event_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Rebalancer report -> %s", path)
        return str(path)
