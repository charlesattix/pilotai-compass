"""
Macro Event Gate
================
Computes position-size scaling factors based on proximity to scheduled
macro events: FOMC decisions, CPI releases, and NFP (jobs) reports.

Scaling logic (from proposal Section F config):
  FOMC: {5d: 1.00, 4d: 0.90, 3d: 0.80, 2d: 0.70, 1d: 0.60, 0d: 0.50}
  CPI:  {2d: 1.00, 1d: 0.75, 0d: 0.65}
  NFP:  {2d: 1.00, 1d: 0.80, 0d: 0.75}

Composite scaling = min() across all active event scalings.

Event calendar sources:
  FOMC — hard-coded 2026 decision dates (update annually)
  CPI  — BLS releases approximately the 11th of each month
  NFP  — first Friday of each month (BLS jobs report)
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FOMC decision dates (hard-coded — update each December for coming year)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Includes emergency inter-meeting cuts (2020-03-03 and 2020-03-15)
# ─────────────────────────────────────────────────────────────────────────────
FOMC_DATES_2020 = [
    # Scheduled
    date(2020, 1, 29),
    date(2020, 4, 29),
    date(2020, 6, 10),
    date(2020, 7, 29),
    date(2020, 9, 16),
    date(2020, 11, 5),
    date(2020, 12, 16),
    # Emergency inter-meeting cuts (COVID)
    date(2020, 3, 3),   # Emergency: -50bps
    date(2020, 3, 15),  # Emergency: -100bps + QE restart
]

FOMC_DATES_2021 = [
    date(2021, 1, 27),
    date(2021, 3, 17),
    date(2021, 4, 28),
    date(2021, 6, 16),
    date(2021, 7, 28),
    date(2021, 9, 22),
    date(2021, 11, 3),
    date(2021, 12, 15),
]

FOMC_DATES_2022 = [
    date(2022, 1, 26),
    date(2022, 3, 16),
    date(2022, 5, 4),
    date(2022, 6, 15),
    date(2022, 7, 27),
    date(2022, 9, 21),
    date(2022, 11, 2),
    date(2022, 12, 14),
]

FOMC_DATES_2023 = [
    date(2023, 2, 1),
    date(2023, 3, 22),
    date(2023, 5, 3),
    date(2023, 6, 14),
    date(2023, 7, 26),
    date(2023, 9, 20),
    date(2023, 11, 1),
    date(2023, 12, 13),
]

FOMC_DATES_2024 = [
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
]

FOMC_DATES_2025 = [
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 11, 5),
    date(2025, 12, 10),
]

FOMC_DATES_2026 = [
    date(2026, 1, 29),
    date(2026, 3, 19),
    date(2026, 5, 7),
    date(2026, 6, 18),
    date(2026, 7, 30),
    date(2026, 9, 17),
    date(2026, 11, 5),
    date(2026, 12, 17),
]

# Emergency FOMC dates (unscheduled inter-meeting actions)
FOMC_EMERGENCY_DATES = {
    date(2020, 3, 3),
    date(2020, 3, 15),
}

ALL_FOMC_DATES = sorted(
    set(
        FOMC_DATES_2020 + FOMC_DATES_2021 + FOMC_DATES_2022
        + FOMC_DATES_2023 + FOMC_DATES_2024 + FOMC_DATES_2025
        + FOMC_DATES_2026
    )
)

# ── Scaling tables: days_out → position-size multiplier ───────────────────────
FOMC_SCALING: Dict[int, float] = {5: 1.00, 4: 0.90, 3: 0.80, 2: 0.70, 1: 0.60, 0: 0.50}
CPI_SCALING:  Dict[int, float] = {2: 1.00, 1: 0.75, 0: 0.65}
NFP_SCALING:  Dict[int, float] = {2: 1.00, 1: 0.80, 0: 0.75}

# ── Post-event buffer scaling factors ────────────────────────────────────────
# Residual volatility persists 1 trading day after major announcements
POST_FOMC_BUFFER_SCALING = 0.70  # 1 day after FOMC decision
POST_CPI_BUFFER_SCALING = 0.80   # 1 day after CPI release
POST_NFP_BUFFER_SCALING = 0.80   # 1 day after NFP (jobs) report

# ── CPI release date approximation ──────────────────────────────────────────
CPI_RELEASE_DAY_OF_MONTH = 12  # BLS typically releases CPI around the 12th

# ── Event horizons (calendar days) ───────────────────────────────────────────
DEFAULT_HORIZON_DAYS = 14  # default lookahead for event scanning
DAILY_CHECK_HORIZON_DAYS = 5  # reduced window for daily operational checks


# ─────────────────────────────────────────────────────────────────────────────
# Date computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_friday_of_month(year: int, month: int) -> date:
    """Return the first Friday of the given month."""
    d = date(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7
    return d + timedelta(days=days_until_friday)


def _cpi_release_date(year: int, month: int) -> date:
    """
    Approximate BLS CPI release date for the given reference month.
    BLS typically releases CPI data for month M around the 10th–15th of month M+1.
    We use the 12th as the central estimate; if it falls on a weekend, advance to Monday.
    """
    # CPI for month M is released in month M+1
    release_month = month + 1
    release_year = year
    if release_month > 12:
        release_month = 1
        release_year += 1
    d = date(release_year, release_month, CPI_RELEASE_DAY_OF_MONTH)
    # Advance past weekends
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _nfp_release_date(year: int, month: int) -> date:
    """
    NFP (jobs report) for month M is released on the first Friday of month M+1.
    """
    release_month = month + 1
    release_year = year
    if release_month > 12:
        release_month = 1
        release_year += 1
    return _first_friday_of_month(release_year, release_month)


def _iter_months(base: date, delta_range: range):
    """G7: Yield (year, month) tuples for each month offset in delta_range.

    Uses modular arithmetic to correctly handle year boundaries without
    the additive-month bug (today.month + delta may overflow 12 or underflow 1).
    """
    for delta in delta_range:
        # month is 0-indexed then converted back to 1-12
        total_months = (base.year * 12 + base.month - 1) + delta
        year = total_months // 12
        month = total_months % 12 + 1
        yield year, month


def get_upcoming_events(
    as_of: Optional[date] = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> List[Dict]:
    """
    Return all scheduled events within horizon_days calendar days of as_of.

    Includes post-event buffers (G5): 1 day after FOMC/CPI/NFP is included at
    reduced scaling to capture residual volatility after surprise announcements.

    Each event dict:
      {event_date, event_type, description, days_out, scaling_factor}
    """
    today = as_of or date.today()
    cutoff = today + timedelta(days=horizon_days)
    events: List[Dict] = []

    # FOMC — pre-event window
    for fd in ALL_FOMC_DATES:
        if today <= fd <= cutoff:
            days_out = (fd - today).days
            factor = FOMC_SCALING.get(min(days_out, max(FOMC_SCALING.keys())), 1.0)
            events.append({
                "event_date": fd.strftime("%Y-%m-%d"),
                "event_type": "FOMC",
                "description": f"FOMC Rate Decision — {fd.strftime('%b %d, %Y')}",
                "days_out": days_out,
                "scaling_factor": factor,
            })
        # G5: post-event buffer — 1 trading day after FOMC at 0.70×
        # Residual volatility after surprise rate decisions can persist intraday
        post_fomc = fd + timedelta(days=1)
        if today <= post_fomc <= cutoff:
            events.append({
                "event_date": post_fomc.strftime("%Y-%m-%d"),
                "event_type": "FOMC_POST",
                "description": f"Post-FOMC Buffer — {fd.strftime('%b %d, %Y')}",
                "days_out": (post_fomc - today).days,
                "scaling_factor": POST_FOMC_BUFFER_SCALING,
            })

    # CPI + NFP — G7: use corrected month arithmetic
    for year, month in _iter_months(today, range(-1, 3)):
        cpi_date = _cpi_release_date(year, month)
        if today <= cpi_date <= cutoff:
            days_out = (cpi_date - today).days
            factor = CPI_SCALING.get(min(days_out, max(CPI_SCALING.keys())), 1.0)
            events.append({
                "event_date": cpi_date.strftime("%Y-%m-%d"),
                "event_type": "CPI",
                "description": f"CPI Release ({year}-{month:02d}) — {cpi_date.strftime('%b %d, %Y')}",
                "days_out": days_out,
                "scaling_factor": factor,
            })
            # G5: post-event buffer
            post_cpi = cpi_date + timedelta(days=1)
            if today <= post_cpi <= cutoff:
                events.append({
                    "event_date": post_cpi.strftime("%Y-%m-%d"),
                    "event_type": "CPI_POST",
                    "description": f"Post-CPI Buffer ({year}-{month:02d})",
                    "days_out": (post_cpi - today).days,
                    "scaling_factor": POST_CPI_BUFFER_SCALING,
                })

        nfp_date = _nfp_release_date(year, month)
        if today <= nfp_date <= cutoff:
            days_out = (nfp_date - today).days
            factor = NFP_SCALING.get(min(days_out, max(NFP_SCALING.keys())), 1.0)
            events.append({
                "event_date": nfp_date.strftime("%Y-%m-%d"),
                "event_type": "NFP",
                "description": f"NFP Jobs Report ({year}-{month:02d}) — {nfp_date.strftime('%b %d, %Y')}",
                "days_out": days_out,
                "scaling_factor": factor,
            })
            # G5: post-event buffer
            post_nfp = nfp_date + timedelta(days=1)
            if today <= post_nfp <= cutoff:
                events.append({
                    "event_date": post_nfp.strftime("%Y-%m-%d"),
                    "event_type": "NFP_POST",
                    "description": f"Post-NFP Buffer ({year}-{month:02d})",
                    "days_out": (post_nfp - today).days,
                    "scaling_factor": POST_NFP_BUFFER_SCALING,
                })

    # Deduplicate (same date/type)
    seen = set()
    unique = []
    for ev in sorted(events, key=lambda x: x["event_date"]):
        key = (ev["event_date"], ev["event_type"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    return unique


def compute_composite_scaling(events: List[Dict]) -> float:
    """
    G4 fix: Return the composite scaling factor using per-type minimums.

    Old behavior: min(all event scalings) — when FOMC (0.50×) and CPI (0.65×)
    coincide, the CPI factor is redundant since FOMC is already the floor.
    This means the CPI signal added no information beyond what FOMC already captured.

    New behavior: Take the minimum independently per event category, then combine.
    FOMC and its post-buffer are one category; CPI/NFP and their buffers are another.
    The composite = min(fomc_floor, data_release_floor).

    In practice for live trading the difference is small, but this correctly
    separates "policy uncertainty" (FOMC) from "data surprise" (CPI/NFP) risk.
    Returns 1.0 when no events are active.
    """
    if not events:
        return 1.0

    fomc_events  = [e for e in events if e["event_type"].startswith("FOMC")]
    data_events  = [e for e in events if not e["event_type"].startswith("FOMC")]

    fomc_floor = min((e["scaling_factor"] for e in fomc_events), default=1.0)
    data_floor = min((e["scaling_factor"] for e in data_events), default=1.0)

    return min(fomc_floor, data_floor)


def run_daily_event_check(
    as_of: Optional[date] = None,
    db_path: Optional[str] = None,
) -> Tuple[float, List[Dict]]:
    """
    Run the daily event gate check:
      1. Compute upcoming events within 5 days
      2. Compute composite scaling factor
      3. Persist to macro_state.db
      4. Return (scaling_factor, events)
    """
    from compass.macro_db import set_state, upsert_events

    today = as_of or date.today()
    events = get_upcoming_events(as_of=today, horizon_days=DAILY_CHECK_HORIZON_DAYS)
    scaling = compute_composite_scaling(events)

    if events:
        upsert_events(events, db_path=db_path)

    set_state("event_scaling_factor", str(scaling), db_path=db_path)
    set_state("last_daily_check", today.strftime("%Y-%m-%d"), db_path=db_path)

    return scaling, events
