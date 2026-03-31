"""
Event calendar engine for systematic event trading.

Manages earnings, FOMC, CPI, NFP, VIX expiry, OpEx, and quad-witching
dates.  Computes pre-event positioning rules (IV expansion windows,
gamma scalping), post-event mean-reversion detection, event clustering
analysis, historical event P&L database, and optimal entry/exit timing
per event type.

Generates an HTML report at reports/event_calendar.html with calendar
view, historical event P&L, and upcoming events.

Usage::

    from compass.event_calendar import EventCalendarEngine
    engine = EventCalendarEngine(trades_df, events=events_list)
    results = engine.analyze()
    engine.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "event_calendar.html"

EVENT_TYPES = (
    "earnings", "fomc", "cpi", "nfp", "vix_expiry", "opex", "quad_witching",
)

# ── Well-known event generators ─────────────────────────────────────────

def _third_friday(year: int, month: int) -> date:
    """Third Friday of a given month (OpEx)."""
    d = date(year, month, 1)
    # advance to first Friday
    offset = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=offset)
    return first_fri + timedelta(weeks=2)


def generate_opex_dates(year: int) -> List[date]:
    """Monthly OpEx (third Friday) for a year."""
    return [_third_friday(year, m) for m in range(1, 13)]


def generate_quad_witching(year: int) -> List[date]:
    """Quad witching: third Friday of Mar, Jun, Sep, Dec."""
    return [_third_friday(year, m) for m in (3, 6, 9, 12)]


def generate_vix_expiry(year: int) -> List[date]:
    """VIX expiry: typically 30 days before OpEx (Wednesday before 3rd Fri - 30d)."""
    results = []
    for m in range(1, 13):
        opex = _third_friday(year, m)
        vix_exp = opex - timedelta(days=30)
        # Adjust to Wednesday
        while vix_exp.weekday() != 2:
            vix_exp += timedelta(days=1)
        results.append(vix_exp)
    return results


def generate_nfp_dates(year: int) -> List[date]:
    """First Friday of each month (Non-Farm Payrolls)."""
    results = []
    for m in range(1, 13):
        d = date(year, m, 1)
        offset = (4 - d.weekday()) % 7
        results.append(d + timedelta(days=offset))
    return results


def generate_cpi_dates(year: int) -> List[date]:
    """Approximate CPI release: ~10th-14th of each month."""
    results = []
    for m in range(1, 13):
        d = date(year, m, 12)
        # Shift to weekday
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        results.append(d)
    return results


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class Event:
    """A single calendar event."""
    date: date
    event_type: str
    name: str
    description: str = ""


@dataclass
class EventPnL:
    """Historical P&L around a specific event."""
    event: Event
    pre_pnl: float          # P&L from entry to event
    post_pnl: float         # P&L from event to exit
    total_pnl: float
    iv_change: float         # IV change around event
    entry_days_before: int
    exit_days_after: int
    outcome: str             # "win", "loss"


@dataclass
class PreEventRule:
    """Positioning rule for a specific event type."""
    event_type: str
    optimal_entry_days: int   # days before event to enter
    position_scaling: float   # 0-1 fraction of normal size
    iv_expansion_expected: float
    gamma_scalp_window: bool
    description: str


@dataclass
class PostEventSignal:
    """Post-event mean-reversion signal."""
    event: Event
    move_pct: float           # actual move on event day
    mean_reversion_prob: float
    expected_reversal_pct: float
    signal: str               # "fade", "follow", "neutral"
    confidence: float


@dataclass
class EventCluster:
    """Cluster of multiple events in a short window."""
    start_date: date
    end_date: date
    events: List[Event]
    n_events: int
    risk_level: str           # "low", "medium", "high"
    sizing_adjustment: float  # multiplier on position size


@dataclass
class EventTypeStats:
    """Historical statistics for one event type."""
    event_type: str
    n_events: int
    avg_pnl: float
    win_rate: float
    avg_iv_change: float
    best_entry_days: int
    best_exit_days: int
    avg_pre_pnl: float
    avg_post_pnl: float


@dataclass
class UpcomingEvent:
    """An upcoming event with positioning guidance."""
    event: Event
    days_until: int
    rule: Optional[PreEventRule]
    cluster_risk: str


# ── Engine ──────────────────────────────────────────────────────────────


class EventCalendarEngine:
    """Systematic event calendar analysis engine."""

    def __init__(
        self,
        trades: Optional[pd.DataFrame] = None,
        events: Optional[List[Event]] = None,
        years: Optional[List[int]] = None,
        reference_date: Optional[date] = None,
        cluster_window: int = 5,
        pre_event_window: int = 5,
        post_event_window: int = 3,
    ) -> None:
        self.trades = trades.copy() if trades is not None else pd.DataFrame()
        self.reference_date = reference_date or date.today()
        self.cluster_window = cluster_window
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window

        # Build events
        years = years or [self.reference_date.year - 1, self.reference_date.year, self.reference_date.year + 1]
        self.events = list(events or [])
        if not self.events:
            self.events = self._generate_calendar(years)

        # Results
        self.event_pnl: List[EventPnL] = []
        self.pre_event_rules: Dict[str, PreEventRule] = {}
        self.post_event_signals: List[PostEventSignal] = []
        self.clusters: List[EventCluster] = []
        self.type_stats: List[EventTypeStats] = []
        self.upcoming: List[UpcomingEvent] = []

    @classmethod
    def from_csv(
        cls, trades_path: Optional[str] = None,
        events_path: Optional[str] = None, **kwargs: Any,
    ) -> "EventCalendarEngine":
        trades = pd.read_csv(trades_path, parse_dates=True) if trades_path else None
        events = None
        if events_path:
            edf = pd.read_csv(events_path, parse_dates=["date"])
            events = [
                Event(date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                      event_type=row.get("event_type", "other"),
                      name=row.get("name", ""),
                      description=row.get("description", ""))
                for _, row in edf.iterrows()
            ]
        return cls(trades=trades, events=events, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.pre_event_rules = self._build_pre_event_rules()
        self.event_pnl = self._compute_event_pnl()
        self.type_stats = self._compute_type_stats()
        self.post_event_signals = self._detect_post_event_signals()
        self.clusters = self._detect_clusters()
        self.upcoming = self._get_upcoming()
        return {
            "event_pnl": self.event_pnl,
            "pre_event_rules": self.pre_event_rules,
            "post_event_signals": self.post_event_signals,
            "clusters": self.clusters,
            "type_stats": self.type_stats,
            "upcoming": self.upcoming,
        }

    # ── Calendar generation ─────────────────────────────────────────────

    def _generate_calendar(self, years: List[int]) -> List[Event]:
        events: List[Event] = []
        for y in years:
            for d in generate_opex_dates(y):
                events.append(Event(d, "opex", f"OpEx {d:%b %Y}"))
            for d in generate_quad_witching(y):
                events.append(Event(d, "quad_witching", f"Quad Witching {d:%b %Y}"))
            for d in generate_vix_expiry(y):
                events.append(Event(d, "vix_expiry", f"VIX Expiry {d:%b %Y}"))
            for d in generate_nfp_dates(y):
                events.append(Event(d, "nfp", f"NFP {d:%b %Y}"))
            for d in generate_cpi_dates(y):
                events.append(Event(d, "cpi", f"CPI {d:%b %Y}"))
            # FOMC: ~8 per year, approximate
            for m in (1, 3, 5, 6, 7, 9, 11, 12):
                fd = date(y, m, 15)
                while fd.weekday() != 2:
                    fd += timedelta(days=1)
                events.append(Event(fd, "fomc", f"FOMC {fd:%b %Y}"))
        return sorted(events, key=lambda e: e.date)

    # ── Pre-event positioning rules ─────────────────────────────────────

    def _build_pre_event_rules(self) -> Dict[str, PreEventRule]:
        rules = {
            "fomc": PreEventRule(
                "fomc", optimal_entry_days=5, position_scaling=0.6,
                iv_expansion_expected=0.08, gamma_scalp_window=True,
                description="FOMC: enter 5d before, scale to 60%, expect IV expansion ~8%",
            ),
            "cpi": PreEventRule(
                "cpi", optimal_entry_days=3, position_scaling=0.7,
                iv_expansion_expected=0.05, gamma_scalp_window=False,
                description="CPI: enter 3d before, scale to 70%, moderate IV expansion",
            ),
            "nfp": PreEventRule(
                "nfp", optimal_entry_days=2, position_scaling=0.8,
                iv_expansion_expected=0.04, gamma_scalp_window=False,
                description="NFP: enter 2d before, scale to 80%, slight IV expansion",
            ),
            "vix_expiry": PreEventRule(
                "vix_expiry", optimal_entry_days=3, position_scaling=0.7,
                iv_expansion_expected=0.06, gamma_scalp_window=True,
                description="VIX expiry: gamma pin risk, reduce size 3d before",
            ),
            "opex": PreEventRule(
                "opex", optimal_entry_days=2, position_scaling=0.75,
                iv_expansion_expected=0.03, gamma_scalp_window=True,
                description="OpEx: gamma exposure peaks, moderate size reduction",
            ),
            "quad_witching": PreEventRule(
                "quad_witching", optimal_entry_days=5, position_scaling=0.5,
                iv_expansion_expected=0.10, gamma_scalp_window=True,
                description="Quad witching: maximum gamma risk, halve positions 5d before",
            ),
            "earnings": PreEventRule(
                "earnings", optimal_entry_days=3, position_scaling=0.5,
                iv_expansion_expected=0.15, gamma_scalp_window=False,
                description="Earnings: high IV crush risk, halve positions 3d before",
            ),
        }

        # Override from historical data if available
        if len(self.trades) > 10:
            for etype in EVENT_TYPES:
                pnls = self._get_event_pnls_for_type(etype)
                if len(pnls) >= 5 and etype in rules:
                    # Find best entry timing
                    best_days = self._optimal_entry_timing(etype)
                    if best_days is not None:
                        rules[etype].optimal_entry_days = best_days

        return rules

    # ── Event P&L computation ───────────────────────────────────────────

    def _compute_event_pnl(self) -> List[EventPnL]:
        if self.trades.empty:
            return []

        results: List[EventPnL] = []
        pnl_col = "pnl" if "pnl" in self.trades.columns else None
        if pnl_col is None:
            return []

        date_col = None
        for c in ("entry_date", "date", "exit_date"):
            if c in self.trades.columns:
                date_col = c
                break
        if date_col is None:
            return []

        trade_dates = pd.to_datetime(self.trades[date_col])
        trade_pnls = self.trades[pnl_col].values
        iv_col = "iv_change" if "iv_change" in self.trades.columns else None

        for event in self.events:
            ed = pd.Timestamp(event.date)
            # Find trades within pre/post window
            pre_mask = (trade_dates >= ed - pd.Timedelta(days=self.pre_event_window)) & (trade_dates < ed)
            post_mask = (trade_dates >= ed) & (trade_dates <= ed + pd.Timedelta(days=self.post_event_window))

            pre_pnl = float(trade_pnls[pre_mask].sum()) if pre_mask.any() else 0.0
            post_pnl = float(trade_pnls[post_mask].sum()) if post_mask.any() else 0.0
            total = pre_pnl + post_pnl

            if pre_mask.any() or post_mask.any():
                iv_chg = 0.0
                if iv_col:
                    combined = pre_mask | post_mask
                    iv_chg = float(self.trades.loc[combined, iv_col].mean()) if combined.any() else 0.0

                results.append(EventPnL(
                    event=event, pre_pnl=pre_pnl, post_pnl=post_pnl,
                    total_pnl=total, iv_change=iv_chg,
                    entry_days_before=self.pre_event_window,
                    exit_days_after=self.post_event_window,
                    outcome="win" if total > 0 else "loss",
                ))

        return results

    def _get_event_pnls_for_type(self, event_type: str) -> List[float]:
        return [ep.total_pnl for ep in self.event_pnl if ep.event.event_type == event_type]

    # ── Optimal entry timing ────────────────────────────────────────────

    def _optimal_entry_timing(self, event_type: str) -> Optional[int]:
        """Find best days-before-event to enter."""
        if self.trades.empty or "entry_date" not in self.trades.columns:
            return None

        best_days = None
        best_pnl = -float("inf")
        trade_dates = pd.to_datetime(self.trades["entry_date"])
        pnls = self.trades["pnl"].values

        type_events = [e for e in self.events if e.event_type == event_type]
        for days in range(1, 8):
            total = 0.0
            count = 0
            for event in type_events:
                ed = pd.Timestamp(event.date)
                mask = (trade_dates >= ed - pd.Timedelta(days=days)) & \
                       (trade_dates <= ed + pd.Timedelta(days=self.post_event_window))
                if mask.any():
                    total += float(pnls[mask].sum())
                    count += 1
            avg = total / max(count, 1)
            if avg > best_pnl:
                best_pnl = avg
                best_days = days

        return best_days

    # ── Type statistics ─────────────────────────────────────────────────

    def _compute_type_stats(self) -> List[EventTypeStats]:
        if not self.event_pnl:
            return []

        by_type: Dict[str, List[EventPnL]] = defaultdict(list)
        for ep in self.event_pnl:
            by_type[ep.event.event_type].append(ep)

        results: List[EventTypeStats] = []
        for etype, eps in sorted(by_type.items()):
            pnls = [ep.total_pnl for ep in eps]
            wins = sum(1 for p in pnls if p > 0)
            rule = self.pre_event_rules.get(etype)

            results.append(EventTypeStats(
                event_type=etype, n_events=len(eps),
                avg_pnl=float(np.mean(pnls)),
                win_rate=wins / len(eps),
                avg_iv_change=float(np.mean([ep.iv_change for ep in eps])),
                best_entry_days=rule.optimal_entry_days if rule else 3,
                best_exit_days=self.post_event_window,
                avg_pre_pnl=float(np.mean([ep.pre_pnl for ep in eps])),
                avg_post_pnl=float(np.mean([ep.post_pnl for ep in eps])),
            ))
        return sorted(results, key=lambda s: -s.avg_pnl)

    # ── Post-event mean reversion ───────────────────────────────────────

    def _detect_post_event_signals(self) -> List[PostEventSignal]:
        signals: List[PostEventSignal] = []
        if not self.event_pnl:
            return signals

        # Use historical stats to estimate mean-reversion probability
        type_wr: Dict[str, float] = {}
        for ts in self.type_stats:
            type_wr[ts.event_type] = ts.win_rate

        for ep in self.event_pnl[-20:]:
            move = ep.total_pnl
            wr = type_wr.get(ep.event.event_type, 0.5)
            # Mean reversion: large moves tend to reverse
            mr_prob = min(abs(move) / (abs(move) + 100), 0.8)
            expected_rev = -move * mr_prob * 0.3

            if mr_prob > 0.5 and abs(move) > 50:
                signal = "fade"
            elif mr_prob < 0.3:
                signal = "follow"
            else:
                signal = "neutral"

            signals.append(PostEventSignal(
                event=ep.event, move_pct=move,
                mean_reversion_prob=mr_prob,
                expected_reversal_pct=expected_rev,
                signal=signal, confidence=mr_prob,
            ))
        return signals

    # ── Event clustering ────────────────────────────────────────────────

    def _detect_clusters(self) -> List[EventCluster]:
        if not self.events:
            return []

        clusters: List[EventCluster] = []
        sorted_events = sorted(self.events, key=lambda e: e.date)

        i = 0
        while i < len(sorted_events):
            cluster_events = [sorted_events[i]]
            j = i + 1
            while j < len(sorted_events):
                if (sorted_events[j].date - sorted_events[i].date).days <= self.cluster_window:
                    cluster_events.append(sorted_events[j])
                    j += 1
                else:
                    break

            if len(cluster_events) >= 2:
                n = len(cluster_events)
                risk = "high" if n >= 4 else "medium" if n >= 3 else "low"
                sizing = 0.4 if n >= 4 else 0.6 if n >= 3 else 0.8

                clusters.append(EventCluster(
                    start_date=cluster_events[0].date,
                    end_date=cluster_events[-1].date,
                    events=cluster_events, n_events=n,
                    risk_level=risk, sizing_adjustment=sizing,
                ))
            i = j if j > i + 1 else i + 1

        return clusters

    # ── Upcoming events ─────────────────────────────────────────────────

    def _get_upcoming(self, horizon: int = 30) -> List[UpcomingEvent]:
        results: List[UpcomingEvent] = []
        cutoff = self.reference_date + timedelta(days=horizon)

        # Build cluster risk map
        date_risk: Dict[date, str] = {}
        for cl in self.clusters:
            for ev in cl.events:
                date_risk[ev.date] = cl.risk_level

        for event in self.events:
            if self.reference_date <= event.date <= cutoff:
                days = (event.date - self.reference_date).days
                rule = self.pre_event_rules.get(event.event_type)
                cr = date_risk.get(event.date, "low")
                results.append(UpcomingEvent(
                    event=event, days_until=days,
                    rule=rule, cluster_risk=cr,
                ))
        return sorted(results, key=lambda u: u.days_until)

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if not self.pre_event_rules:
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
        return {
            "type_pnl": self._chart_type_pnl(),
            "calendar": self._chart_calendar(),
            "cluster": self._chart_clusters(),
        }

    def _chart_type_pnl(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.type_stats:
            return ""
        names = [s.event_type for s in self.type_stats]
        pnls = [s.avg_pnl for s in self.type_stats]
        wrs = [s.win_rate for s in self.type_stats]
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in pnls]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.barh(names, pnls, color=colors, alpha=0.85)
        ax1.set_xlabel("Avg P&L ($)"); ax1.set_title("Avg P&L by Event Type", fontsize=10)
        ax1.axvline(0, color="black", lw=0.5); ax1.grid(True, axis="x", alpha=0.3)
        ax2.barh(names, wrs, color="#3b82f6", alpha=0.85)
        ax2.set_xlabel("Win Rate"); ax2.set_title("Win Rate by Event Type", fontsize=10)
        ax2.set_xlim(0, 1); ax2.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_calendar(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        upcoming = self.upcoming[:30]
        if not upcoming:
            return ""
        fig, ax = plt.subplots(figsize=(10, max(3, len(upcoming) * 0.3)))
        type_colors = {
            "fomc": "#dc2626", "cpi": "#f59e0b", "nfp": "#3b82f6",
            "opex": "#16a34a", "quad_witching": "#7f1d1d",
            "vix_expiry": "#8b5cf6", "earnings": "#06b6d4",
        }
        for i, u in enumerate(upcoming):
            c = type_colors.get(u.event.event_type, "#64748b")
            ax.barh(i, u.days_until, color=c, alpha=0.85, height=0.7)
            ax.text(u.days_until + 0.5, i, f"{u.event.name}", va="center", fontsize=7)
        ax.set_yticks(range(len(upcoming)))
        ax.set_yticklabels([f"{u.event.date}" for u in upcoming], fontsize=7)
        ax.set_xlabel("Days Until Event"); ax.set_title("Upcoming Events", fontsize=11)
        ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_clusters(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.clusters:
            return ""
        recent = [c for c in self.clusters
                  if c.end_date >= self.reference_date - timedelta(days=60)][:20]
        if not recent:
            return ""
        fig, ax = plt.subplots(figsize=(8, max(3, len(recent) * 0.4)))
        risk_colors = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}
        for i, cl in enumerate(recent):
            span = (cl.end_date - cl.start_date).days + 1
            c = risk_colors.get(cl.risk_level, "#64748b")
            ax.barh(i, span, left=0, color=c, alpha=0.85, height=0.6)
            ax.text(span + 0.3, i, f"{cl.n_events} events ({cl.risk_level})", va="center", fontsize=7)
        ax.set_yticks(range(len(recent)))
        ax.set_yticklabels([f"{cl.start_date}" for cl in recent], fontsize=7)
        ax.set_xlabel("Cluster Span (days)"); ax.set_title("Event Clusters", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_upcoming = len(self.upcoming)
        n_clusters = len([c for c in self.clusters if c.end_date >= self.reference_date])

        # Upcoming events table
        up_rows = ""
        for u in self.upcoming[:25]:
            rule = u.rule
            scaling = f"{rule.position_scaling:.0%}" if rule else "100%"
            gamma = "Yes" if rule and rule.gamma_scalp_window else "No"
            risk_cls = {"high": "bad", "medium": "warn"}.get(u.cluster_risk, "good")
            up_rows += (
                f'<tr><td>{u.event.date}</td><td>{u.days_until}d</td>'
                f'<td>{u.event.event_type}</td><td>{u.event.name}</td>'
                f'<td>{scaling}</td><td>{gamma}</td>'
                f'<td class="{risk_cls}">{u.cluster_risk}</td></tr>\n'
            )
        if not up_rows:
            up_rows = '<tr><td colspan="7" style="text-align:center;color:#64748b">No upcoming events</td></tr>'

        # Type stats table
        stat_rows = ""
        for s in self.type_stats:
            cls = "good" if s.avg_pnl > 0 else "bad"
            stat_rows += (
                f'<tr><td>{s.event_type}</td><td>{s.n_events}</td>'
                f'<td class="{cls}">${s.avg_pnl:,.0f}</td>'
                f'<td>{s.win_rate:.0%}</td><td>{s.avg_iv_change:+.1%}</td>'
                f'<td>{s.best_entry_days}d</td><td>{s.best_exit_days}d</td></tr>\n'
            )

        # Rules table
        rule_rows = ""
        for etype in EVENT_TYPES:
            r = self.pre_event_rules.get(etype)
            if r:
                rule_rows += (
                    f'<tr><td>{r.event_type}</td><td>{r.optimal_entry_days}d</td>'
                    f'<td>{r.position_scaling:.0%}</td>'
                    f'<td>{r.iv_expansion_expected:+.0%}</td>'
                    f'<td>{"Yes" if r.gamma_scalp_window else "No"}</td>'
                    f'<td style="text-align:left">{r.description}</td></tr>\n'
                )

        # Cluster table
        cluster_rows = ""
        for cl in self.clusters[:15]:
            risk_cls = {"high": "bad", "medium": "warn"}.get(cl.risk_level, "")
            evts = ", ".join(e.event_type for e in cl.events[:4])
            cluster_rows += (
                f'<tr><td>{cl.start_date} – {cl.end_date}</td>'
                f'<td>{cl.n_events}</td><td class="{risk_cls}">{cl.risk_level}</td>'
                f'<td>{cl.sizing_adjustment:.0%}</td><td>{evts}</td></tr>\n'
            )
        if not cluster_rows:
            cluster_rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">No clusters</td></tr>'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Event Calendar Dashboard</title>
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
<h1>Event Calendar Dashboard</h1>
<div class="meta">{len(self.events)} events &middot; {len(self.event_pnl)} with P&L &middot; Reference: {self.reference_date} &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value">{n_upcoming}</div><div class="label">Upcoming (30d)</div></div>
  <div class="kpi"><div class="value">{n_clusters}</div><div class="label">Active Clusters</div></div>
  <div class="kpi"><div class="value">{len(self.type_stats)}</div><div class="label">Event Types Tracked</div></div>
  <div class="kpi"><div class="value">{len(self.event_pnl)}</div><div class="label">Historical Events</div></div>
</div>
<h2>1. Upcoming Events</h2>{_img("calendar")}
<table><thead><tr><th>Date</th><th>Days</th><th>Type</th><th>Name</th><th>Sizing</th><th>Gamma</th><th>Cluster Risk</th></tr></thead>
<tbody>{up_rows}</tbody></table>
<h2>2. Pre-Event Positioning Rules</h2>
<table><thead><tr><th>Type</th><th>Entry</th><th>Size</th><th>IV Exp</th><th>Gamma</th><th>Description</th></tr></thead>
<tbody>{rule_rows}</tbody></table>
<h2>3. Historical Event P&L</h2>{_img("type_pnl")}
<table><thead><tr><th>Type</th><th>Events</th><th>Avg P&L</th><th>Win Rate</th><th>Avg IV Chg</th><th>Best Entry</th><th>Best Exit</th></tr></thead>
<tbody>{stat_rows if stat_rows else '<tr><td colspan="7" style="text-align:center;color:#64748b">No trade data</td></tr>'}</tbody></table>
<h2>4. Event Clusters</h2>{_img("cluster")}
<table><thead><tr><th>Window</th><th>Events</th><th>Risk</th><th>Sizing</th><th>Types</th></tr></thead>
<tbody>{cluster_rows}</tbody></table>
<footer>Generated by <code>compass/event_calendar.py</code></footer>
</body></html>"""
        return html
