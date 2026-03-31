"""Event impact analyzer – measures how macro events affect credit spread outcomes.

Supports FOMC, CPI, NFP, VIX expiration, and OPEX events.  Decomposes P&L into
pre-event and post-event windows, measures IV crush, and identifies optimal entry
timing relative to each event type.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Event type constants ────────────────────────────────────────────────────
EVENT_TYPES: List[str] = ["FOMC", "CPI", "NFP", "VIX_EXP", "OPEX"]

# Pre/post windows in calendar days
PRE_WINDOW_DAYS: int = 5
POST_WINDOW_DAYS: int = 5

# Default entry offsets to evaluate (days before event)
DEFAULT_ENTRY_OFFSETS: List[int] = [0, 1, 2, 3, 5, 7, 10]

# Expected IV crush by event type (baseline estimates)
BASELINE_IV_CRUSH: Dict[str, float] = {
    "FOMC": 0.50,
    "CPI": 0.40,
    "NFP": 0.35,
    "VIX_EXP": 0.25,
    "OPEX": 0.20,
}


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class EventWindow:
    """A single event occurrence with surrounding trade data."""
    event_type: str
    event_date: date
    pre_pnl: float         # cumulative P&L in pre-event window
    post_pnl: float        # cumulative P&L in post-event window
    total_pnl: float       # pre + post
    pre_iv: float          # average IV before event
    post_iv: float         # average IV after event
    iv_crush_pct: float    # (pre_iv - post_iv) / pre_iv
    pre_return: float      # underlying return pre-event
    post_return: float     # underlying return post-event
    n_trades: int          # trades in window


@dataclass
class EventTypeStats:
    """Aggregated statistics for one event type."""
    event_type: str
    n_events: int
    win_rate: float
    avg_pnl: float
    median_pnl: float
    avg_pre_pnl: float
    avg_post_pnl: float
    avg_iv_crush_pct: float
    std_iv_crush_pct: float
    avg_pre_iv: float
    avg_post_iv: float
    best_entry_offset: int       # days before event for best avg P&L
    best_entry_avg_pnl: float
    pnl_by_offset: Dict[int, float]


@dataclass
class TimingResult:
    """Optimal entry timing for a single event type."""
    event_type: str
    offset_days: int
    avg_pnl: float
    win_rate: float
    n_trades: int


@dataclass
class IVCrushResult:
    """IV crush measurement for a single event type."""
    event_type: str
    avg_crush_pct: float
    median_crush_pct: float
    std_crush_pct: float
    crush_vs_baseline: float  # actual / expected
    n_events: int


@dataclass
class EventImpactResult:
    """Full event impact analysis."""
    event_stats: List[EventTypeStats] = field(default_factory=list)
    timing_results: List[TimingResult] = field(default_factory=list)
    iv_crush_results: List[IVCrushResult] = field(default_factory=list)
    event_windows: List[EventWindow] = field(default_factory=list)
    generated_at: str = ""


# ── Event calendar helpers ──────────────────────────────────────────────────
def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of a given month (OPEX day)."""
    first = date(year, month, 1)
    # weekday(): Monday=0 … Friday=4
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def _third_wednesday(year: int, month: int) -> date:
    """Return the third Wednesday of a given month (VIX expiration is 30 days
    before next month's third Friday, but approximated as third Wednesday)."""
    first = date(year, month, 1)
    days_to_wed = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=days_to_wed)
    return first_wed + timedelta(weeks=2)


def _first_friday(year: int, month: int) -> date:
    """First Friday of month (NFP release)."""
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday)


def _cpi_date(year: int, month: int) -> date:
    """Approximate CPI release: second Wednesday of month."""
    first = date(year, month, 1)
    days_to_wed = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=days_to_wed)
    return first_wed + timedelta(weeks=1)


def build_event_calendar(
    start: date,
    end: date,
    fomc_dates: Optional[Sequence[date]] = None,
) -> pd.DataFrame:
    """Build a DataFrame of event dates between start and end.

    Parameters
    ----------
    start, end : date
        Date range (inclusive).
    fomc_dates : sequence of date, optional
        Hard-coded FOMC dates.  If *None*, attempts to import from
        ``shared.constants`` or ``compass.events``.

    Returns
    -------
    pd.DataFrame with columns [event_type, event_date].
    """
    rows: List[Dict[str, object]] = []

    # -- FOMC --
    if fomc_dates is None:
        fomc_dates = _try_import_fomc_dates()
    if fomc_dates is not None:
        for d in fomc_dates:
            if isinstance(d, datetime):
                d = d.date()
            if start <= d <= end:
                rows.append({"event_type": "FOMC", "event_date": d})

    # -- Monthly algorithmics --
    cur = date(start.year, start.month, 1)
    while cur <= end:
        y, m = cur.year, cur.month

        cpi = _cpi_date(y, m)
        if start <= cpi <= end:
            rows.append({"event_type": "CPI", "event_date": cpi})

        nfp = _first_friday(y, m)
        if start <= nfp <= end:
            rows.append({"event_type": "NFP", "event_date": nfp})

        vix = _third_wednesday(y, m)
        if start <= vix <= end:
            rows.append({"event_type": "VIX_EXP", "event_date": vix})

        opex = _third_friday(y, m)
        if start <= opex <= end:
            rows.append({"event_type": "OPEX", "event_date": opex})

        # advance month
        if m == 12:
            cur = date(y + 1, 1, 1)
        else:
            cur = date(y, m + 1, 1)

    if not rows:
        return pd.DataFrame(columns=["event_type", "event_date"])

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df.sort_values("event_date").reset_index(drop=True)


def _try_import_fomc_dates() -> Optional[List[date]]:
    """Best-effort import of hard-coded FOMC dates."""
    try:
        from shared.constants import FOMC_DATES  # type: ignore[import-untyped]
        return [d.date() if isinstance(d, datetime) else d for d in FOMC_DATES]
    except Exception:
        pass
    try:
        from compass.events import FOMC_DATES as FD  # type: ignore[import-untyped]
        return [d if isinstance(d, date) else d.date() for d in FD]
    except Exception:
        pass
    return None


# ── Core analyzer ───────────────────────────────────────────────────────────
class EventImpactAnalyzer:
    """Measures how macro events affect credit spread outcomes."""

    def __init__(
        self,
        pre_window: int = PRE_WINDOW_DAYS,
        post_window: int = POST_WINDOW_DAYS,
        entry_offsets: Optional[List[int]] = None,
    ) -> None:
        self.pre_window = pre_window
        self.post_window = post_window
        self.entry_offsets = entry_offsets or list(DEFAULT_ENTRY_OFFSETS)

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        trades: pd.DataFrame,
        iv_series: Optional[pd.Series] = None,
        price_series: Optional[pd.Series] = None,
        event_calendar: Optional[pd.DataFrame] = None,
    ) -> EventImpactResult:
        """Run full event impact analysis.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have columns: date (or entry_date), pnl.
            Optional: iv, entry_offset, event_type.
        iv_series : pd.Series, optional
            Daily IV (e.g. VIX) indexed by date.
        price_series : pd.Series, optional
            Daily underlying price indexed by date.
        event_calendar : pd.DataFrame, optional
            Columns [event_type, event_date].  Built automatically if None.
        """
        trades = self._normalize_trades(trades)
        if trades.empty:
            return EventImpactResult(generated_at=self._now())

        date_range = (trades["date"].min(), trades["date"].max())
        if event_calendar is None:
            event_calendar = build_event_calendar(date_range[0], date_range[1])

        if event_calendar.empty:
            logger.warning("No events in date range %s – %s", *date_range)
            return EventImpactResult(generated_at=self._now())

        windows = self._build_event_windows(trades, event_calendar, iv_series, price_series)
        event_stats = self._compute_event_stats(windows, trades, event_calendar)
        timing = self._compute_timing(trades, event_calendar)
        iv_crush = self._compute_iv_crush(windows)

        return EventImpactResult(
            event_stats=event_stats,
            timing_results=timing,
            iv_crush_results=iv_crush,
            event_windows=windows,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: EventImpactResult,
        output_path: str | Path = "reports/event_impact.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Event impact report written to %s", path)
        return path

    # ── Trade normalization ─────────────────────────────────────────────────
    @staticmethod
    def _normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
        df = trades.copy()
        if "entry_date" in df.columns and "date" not in df.columns:
            df["date"] = df["entry_date"]
        if "date" not in df.columns or "pnl" not in df.columns:
            logger.error("trades must have 'date' and 'pnl' columns")
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ── Event windows ───────────────────────────────────────────────────────
    def _build_event_windows(
        self,
        trades: pd.DataFrame,
        calendar: pd.DataFrame,
        iv_series: Optional[pd.Series],
        price_series: Optional[pd.Series],
    ) -> List[EventWindow]:
        windows: List[EventWindow] = []
        for _, row in calendar.iterrows():
            etype = row["event_type"]
            edate = row["event_date"]
            if isinstance(edate, datetime):
                edate = edate.date()

            pre_start = edate - timedelta(days=self.pre_window)
            post_end = edate + timedelta(days=self.post_window)

            pre_trades = trades[(trades["date"] >= pre_start) & (trades["date"] < edate)]
            post_trades = trades[(trades["date"] >= edate) & (trades["date"] <= post_end)]

            pre_pnl = float(pre_trades["pnl"].sum()) if not pre_trades.empty else 0.0
            post_pnl = float(post_trades["pnl"].sum()) if not post_trades.empty else 0.0
            n_trades = len(pre_trades) + len(post_trades)

            pre_iv, post_iv, iv_crush = self._extract_iv(edate, iv_series)
            pre_ret, post_ret = self._extract_returns(edate, price_series)

            windows.append(EventWindow(
                event_type=etype,
                event_date=edate,
                pre_pnl=pre_pnl,
                post_pnl=post_pnl,
                total_pnl=pre_pnl + post_pnl,
                pre_iv=pre_iv,
                post_iv=post_iv,
                iv_crush_pct=iv_crush,
                pre_return=pre_ret,
                post_return=post_ret,
                n_trades=n_trades,
            ))
        return windows

    def _extract_iv(
        self, edate: date, iv_series: Optional[pd.Series],
    ) -> Tuple[float, float, float]:
        if iv_series is None or iv_series.empty:
            return (0.0, 0.0, 0.0)
        idx = iv_series.index
        if hasattr(idx[0], "date"):
            dates = pd.Series([d.date() if hasattr(d, "date") else d for d in idx])
        else:
            dates = pd.Series(idx)

        pre_mask = (dates >= (edate - timedelta(days=self.pre_window))) & (dates < edate)
        post_mask = (dates >= edate) & (dates <= (edate + timedelta(days=self.post_window)))

        pre_vals = iv_series.iloc[pre_mask.values]
        post_vals = iv_series.iloc[post_mask.values]

        pre_iv = float(pre_vals.mean()) if len(pre_vals) > 0 else 0.0
        post_iv = float(post_vals.mean()) if len(post_vals) > 0 else 0.0

        if pre_iv > 1e-9:
            crush = (pre_iv - post_iv) / pre_iv
        else:
            crush = 0.0
        return (pre_iv, post_iv, crush)

    def _extract_returns(
        self, edate: date, price_series: Optional[pd.Series],
    ) -> Tuple[float, float]:
        if price_series is None or price_series.empty:
            return (0.0, 0.0)
        idx = price_series.index
        if hasattr(idx[0], "date"):
            dates = pd.Series([d.date() if hasattr(d, "date") else d for d in idx])
        else:
            dates = pd.Series(idx)

        pre_mask = (dates >= (edate - timedelta(days=self.pre_window))) & (dates < edate)
        post_mask = (dates >= edate) & (dates <= (edate + timedelta(days=self.post_window)))

        pre_prices = price_series.iloc[pre_mask.values]
        post_prices = price_series.iloc[post_mask.values]

        pre_ret = 0.0
        if len(pre_prices) >= 2:
            pre_ret = float((pre_prices.iloc[-1] / pre_prices.iloc[0]) - 1.0)

        post_ret = 0.0
        if len(post_prices) >= 2:
            post_ret = float((post_prices.iloc[-1] / post_prices.iloc[0]) - 1.0)

        return (pre_ret, post_ret)

    # ── Event type stats ────────────────────────────────────────────────────
    def _compute_event_stats(
        self,
        windows: List[EventWindow],
        trades: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> List[EventTypeStats]:
        stats: List[EventTypeStats] = []
        for etype in EVENT_TYPES:
            ew = [w for w in windows if w.event_type == etype]
            if not ew:
                continue

            pnls = np.array([w.total_pnl for w in ew])
            pre_pnls = np.array([w.pre_pnl for w in ew])
            post_pnls = np.array([w.post_pnl for w in ew])
            crushes = np.array([w.iv_crush_pct for w in ew])

            # timing by offset
            pnl_by_offset = self._pnl_by_offset(trades, calendar, etype)
            if pnl_by_offset:
                best_off = max(pnl_by_offset, key=pnl_by_offset.get)
                best_pnl = pnl_by_offset[best_off]
            else:
                best_off = 0
                best_pnl = 0.0

            stats.append(EventTypeStats(
                event_type=etype,
                n_events=len(ew),
                win_rate=float(np.mean(pnls > 0)) if len(pnls) > 0 else 0.0,
                avg_pnl=float(np.mean(pnls)),
                median_pnl=float(np.median(pnls)),
                avg_pre_pnl=float(np.mean(pre_pnls)),
                avg_post_pnl=float(np.mean(post_pnls)),
                avg_iv_crush_pct=float(np.mean(crushes)),
                std_iv_crush_pct=float(np.std(crushes)),
                avg_pre_iv=float(np.mean([w.pre_iv for w in ew])),
                avg_post_iv=float(np.mean([w.post_iv for w in ew])),
                best_entry_offset=best_off,
                best_entry_avg_pnl=best_pnl,
                pnl_by_offset=pnl_by_offset,
            ))
        return stats

    def _pnl_by_offset(
        self, trades: pd.DataFrame, calendar: pd.DataFrame, etype: str,
    ) -> Dict[int, float]:
        event_dates = [
            r["event_date"] for _, r in calendar.iterrows()
            if r["event_type"] == etype
        ]
        result: Dict[int, float] = {}
        for offset in self.entry_offsets:
            pnls: List[float] = []
            for ed in event_dates:
                if isinstance(ed, datetime):
                    ed = ed.date()
                entry = ed - timedelta(days=offset)
                exit_d = ed + timedelta(days=self.post_window)
                window = trades[(trades["date"] >= entry) & (trades["date"] <= exit_d)]
                if not window.empty:
                    pnls.append(float(window["pnl"].sum()))
            if pnls:
                result[offset] = float(np.mean(pnls))
        return result

    # ── Timing analysis ─────────────────────────────────────────────────────
    def _compute_timing(
        self, trades: pd.DataFrame, calendar: pd.DataFrame,
    ) -> List[TimingResult]:
        results: List[TimingResult] = []
        for etype in EVENT_TYPES:
            pnl_by_off = self._pnl_by_offset(trades, calendar, etype)
            if not pnl_by_off:
                continue
            best_off = max(pnl_by_off, key=pnl_by_off.get)
            # count trades at best offset
            event_dates = [
                r["event_date"] for _, r in calendar.iterrows()
                if r["event_type"] == etype
            ]
            n = 0
            wins = 0
            for ed in event_dates:
                if isinstance(ed, datetime):
                    ed = ed.date()
                entry = ed - timedelta(days=best_off)
                exit_d = ed + timedelta(days=self.post_window)
                w = trades[(trades["date"] >= entry) & (trades["date"] <= exit_d)]
                if not w.empty:
                    n += 1
                    if w["pnl"].sum() > 0:
                        wins += 1
            results.append(TimingResult(
                event_type=etype,
                offset_days=best_off,
                avg_pnl=pnl_by_off[best_off],
                win_rate=wins / n if n > 0 else 0.0,
                n_trades=n,
            ))
        return results

    # ── IV crush ────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_iv_crush(windows: List[EventWindow]) -> List[IVCrushResult]:
        results: List[IVCrushResult] = []
        for etype in EVENT_TYPES:
            ew = [w for w in windows if w.event_type == etype and w.pre_iv > 1e-9]
            if not ew:
                continue
            crushes = np.array([w.iv_crush_pct for w in ew])
            baseline = BASELINE_IV_CRUSH.get(etype, 0.0)
            avg_crush = float(np.mean(crushes))
            results.append(IVCrushResult(
                event_type=etype,
                avg_crush_pct=avg_crush,
                median_crush_pct=float(np.median(crushes)),
                std_crush_pct=float(np.std(crushes)),
                crush_vs_baseline=avg_crush / baseline if baseline > 1e-9 else 0.0,
                n_events=len(ew),
            ))
        return results

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: EventImpactResult) -> str:
        summary_cards = self._html_summary_cards(r)
        event_table = self._html_event_table(r.event_stats)
        pnl_bars = self._svg_pnl_bars(r.event_stats)
        timing_section = self._html_timing(r.timing_results)
        iv_section = self._html_iv_crush(r.iv_crush_results)
        decomp_section = self._html_pnl_decomp(r.event_stats)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Event Impact Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.subtitle{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .value{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.section{{margin-bottom:32px}}
.section h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto}}
.bar-label{{font-size:11px;fill:#94a3b8}}
.bar-value{{font-size:11px;fill:#e2e8f0}}
</style>
</head>
<body>
<h1>Event Impact Analysis</h1>
<p class="subtitle">Generated {r.generated_at or 'N/A'}</p>

{summary_cards}

<div class="section">
<h2>P&L by Event Type</h2>
{pnl_bars}
</div>

{event_table}
{decomp_section}
{timing_section}
{iv_section}

</body>
</html>"""

    # ── Summary cards ───────────────────────────────────────────────────────
    @staticmethod
    def _html_summary_cards(r: EventImpactResult) -> str:
        total_events = sum(s.n_events for s in r.event_stats)
        avg_pnl = (
            np.mean([s.avg_pnl for s in r.event_stats])
            if r.event_stats else 0.0
        )
        best = max(r.event_stats, key=lambda s: s.avg_pnl) if r.event_stats else None
        worst = min(r.event_stats, key=lambda s: s.avg_pnl) if r.event_stats else None
        avg_crush = (
            np.mean([c.avg_crush_pct for c in r.iv_crush_results])
            if r.iv_crush_results else 0.0
        )
        return f"""<div class="grid">
<div class="card"><div class="label">Total Events</div><div class="value">{total_events}</div></div>
<div class="card"><div class="label">Avg P&L / Event</div><div class="value">{avg_pnl:.2f}</div></div>
<div class="card"><div class="label">Best Event</div><div class="value">{best.event_type if best else 'N/A'}</div></div>
<div class="card"><div class="label">Worst Event</div><div class="value">{worst.event_type if worst else 'N/A'}</div></div>
<div class="card"><div class="label">Avg IV Crush</div><div class="value">{avg_crush:.1%}</div></div>
</div>"""

    # ── Event stats table ───────────────────────────────────────────────────
    @staticmethod
    def _html_event_table(stats: List[EventTypeStats]) -> str:
        if not stats:
            return ""
        rows = ""
        for s in stats:
            pnl_cls = "pos" if s.avg_pnl >= 0 else "neg"
            rows += (
                f"<tr><td>{s.event_type}</td><td>{s.n_events}</td>"
                f'<td class="{pnl_cls}">{s.avg_pnl:.2f}</td>'
                f"<td>{s.median_pnl:.2f}</td>"
                f"<td>{s.win_rate:.1%}</td>"
                f"<td>{s.avg_iv_crush_pct:.1%}</td>"
                f"<td>{s.best_entry_offset}d</td></tr>"
            )
        return f"""<div class="section">
<h2>Event Type Summary</h2>
<table>
<thead><tr><th>Type</th><th>Events</th><th>Avg P&L</th><th>Med P&L</th><th>Win Rate</th><th>Avg IV Crush</th><th>Best Entry</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── P&L decomposition table ─────────────────────────────────────────────
    @staticmethod
    def _html_pnl_decomp(stats: List[EventTypeStats]) -> str:
        if not stats:
            return ""
        rows = ""
        for s in stats:
            pre_cls = "pos" if s.avg_pre_pnl >= 0 else "neg"
            post_cls = "pos" if s.avg_post_pnl >= 0 else "neg"
            rows += (
                f"<tr><td>{s.event_type}</td>"
                f'<td class="{pre_cls}">{s.avg_pre_pnl:.2f}</td>'
                f'<td class="{post_cls}">{s.avg_post_pnl:.2f}</td>'
                f"<td>{s.avg_pre_iv:.2f}</td>"
                f"<td>{s.avg_post_iv:.2f}</td>"
                f"<td>{s.avg_iv_crush_pct:.1%}</td></tr>"
            )
        return f"""<div class="section">
<h2>Pre/Post Event P&L Decomposition</h2>
<table>
<thead><tr><th>Type</th><th>Avg Pre P&L</th><th>Avg Post P&L</th><th>Avg Pre IV</th><th>Avg Post IV</th><th>IV Crush</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Timing section ──────────────────────────────────────────────────────
    @staticmethod
    def _html_timing(timing: List[TimingResult]) -> str:
        if not timing:
            return ""
        rows = ""
        for t in timing:
            pnl_cls = "pos" if t.avg_pnl >= 0 else "neg"
            rows += (
                f"<tr><td>{t.event_type}</td>"
                f"<td>{t.offset_days}d before</td>"
                f'<td class="{pnl_cls}">{t.avg_pnl:.2f}</td>'
                f"<td>{t.win_rate:.1%}</td>"
                f"<td>{t.n_trades}</td></tr>"
            )
        return f"""<div class="section">
<h2>Optimal Entry Timing</h2>
<table>
<thead><tr><th>Type</th><th>Best Entry</th><th>Avg P&L</th><th>Win Rate</th><th>Trades</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── IV crush section ────────────────────────────────────────────────────
    @staticmethod
    def _html_iv_crush(crushes: List[IVCrushResult]) -> str:
        if not crushes:
            return ""
        rows = ""
        for c in crushes:
            rows += (
                f"<tr><td>{c.event_type}</td>"
                f"<td>{c.avg_crush_pct:.1%}</td>"
                f"<td>{c.median_crush_pct:.1%}</td>"
                f"<td>{c.std_crush_pct:.1%}</td>"
                f"<td>{c.crush_vs_baseline:.2f}x</td>"
                f"<td>{c.n_events}</td></tr>"
            )
        return f"""<div class="section">
<h2>IV Crush by Event Type</h2>
<table>
<thead><tr><th>Type</th><th>Avg Crush</th><th>Median</th><th>Std</th><th>vs Baseline</th><th>Events</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── SVG bar chart ───────────────────────────────────────────────────────
    @staticmethod
    def _svg_pnl_bars(stats: List[EventTypeStats]) -> str:
        if not stats:
            return "<p>No event data available.</p>"
        w, h = 520, 240
        pad_l, pad_b, pad_t = 60, 40, 20
        chart_h = h - pad_b - pad_t
        n = len(stats)
        max_abs = max(abs(s.avg_pnl) for s in stats) or 1.0
        bar_w = min(55, (w - pad_l) // n - 12)
        mid_y = pad_t + chart_h // 2

        bars = ""
        for i, s in enumerate(stats):
            x = pad_l + i * ((w - pad_l) // n) + 10
            scaled = (s.avg_pnl / max_abs) * (chart_h * 0.4)
            if s.avg_pnl >= 0:
                bar_h = scaled
                y = mid_y - bar_h
                colour = "#4ade80"
            else:
                bar_h = -scaled
                y = mid_y
                colour = "#f87171"
            bars += (
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
                f'rx="3" fill="{colour}" opacity="0.85"/>'
                f'<text x="{x + bar_w // 2}" y="{y - 6 if s.avg_pnl >= 0 else y + bar_h + 14}" '
                f'text-anchor="middle" class="bar-value">{s.avg_pnl:.2f}</text>'
                f'<text x="{x + bar_w // 2}" y="{h - 10}" text-anchor="middle" '
                f'class="bar-label">{s.event_type}</text>'
            )

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pad_l}" y1="{mid_y}" x2="{w}" y2="{mid_y}" '
            f'stroke="#475569" stroke-width="1" stroke-dasharray="4"/>'
            f"{bars}</svg>"
        )
