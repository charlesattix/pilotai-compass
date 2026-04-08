"""
Calendar effects alpha engine.

Detects and scores 8 calendar anomalies for SPY options timing:
  1. Turn of Month (ToM)         5. Santa Rally
  2. Options Expiration (OpEx)   6. Sell in May
  3. FOMC Drift                  7. Monday Effect
  4. Quad Witching               8. Month-End Rebalancing

Outputs a daily composite score (-1 to +1) usable standalone
or as a timing overlay for existing strategies.

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Known FOMC dates 2020-2026 (announcement day)
FOMC_DATES: Set[str] = {
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CalendarScore:
    """Daily calendar effect scores."""
    date: datetime
    tom: float = 0.0              # turn of month
    opex: float = 0.0             # options expiration week
    fomc: float = 0.0             # FOMC drift
    quad_witch: float = 0.0       # quad witching
    santa: float = 0.0            # Santa rally
    sell_may: float = 0.0         # sell in May
    monday: float = 0.0           # Monday effect
    month_end: float = 0.0        # month-end rebalancing
    composite: float = 0.0        # weighted aggregate


@dataclass
class EffectStats:
    """Statistical significance of one calendar effect."""
    name: str
    n_days: int
    avg_return: float
    baseline_return: float
    excess_return: float
    t_stat: float
    p_value: float
    significant: bool


@dataclass
class CalendarBacktestResult:
    """Backtest results for calendar timing."""
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_active_days: int
    n_total_days: int
    active_pct: float
    effect_stats: List[EffectStats]


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class CalendarEffects:
    """Calendar effects detection and scoring.

    Args:
        weights: Per-effect weights for composite score.
    """

    DEFAULT_WEIGHTS = {
        "tom": 0.20, "opex": 0.10, "fomc": 0.20, "quad_witch": 0.05,
        "santa": 0.10, "sell_may": 0.10, "monday": 0.10, "month_end": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Individual effect detectors
    # ------------------------------------------------------------------

    @staticmethod
    def is_turn_of_month(dt: pd.Timestamp) -> float:
        """Last 2 + first 3 trading days of month → +1 (bullish)."""
        dom = dt.day
        days_in_month = pd.Timestamp(dt.year, dt.month, 1).days_in_month
        if dom <= 3:
            return 1.0
        if dom >= days_in_month - 2:
            return 1.0
        return 0.0

    @staticmethod
    def is_opex_week(dt: pd.Timestamp) -> float:
        """Third Friday of month ± 2 days → slight negative (higher vol)."""
        # Third Friday: first day of month, advance to first Friday, add 14
        first = pd.Timestamp(dt.year, dt.month, 1)
        first_fri = first + pd.offsets.Week(weekday=4)
        if first_fri.month != dt.month:
            first_fri += pd.Timedelta(days=7)
        third_fri = first_fri + pd.Timedelta(days=14)
        diff = abs((dt - third_fri).days)
        if diff <= 2:
            return -0.3  # slightly bearish (vol expansion)
        return 0.0

    @staticmethod
    def is_fomc_drift(dt: pd.Timestamp) -> float:
        """Day before FOMC → +1 (pre-FOMC drift is bullish)."""
        tomorrow = dt + pd.offsets.BDay(1)
        if tomorrow.strftime("%Y-%m-%d") in FOMC_DATES:
            return 1.0
        if dt.strftime("%Y-%m-%d") in FOMC_DATES:
            return 0.5  # FOMC day itself: moderate positive
        return 0.0

    @staticmethod
    def is_quad_witching(dt: pd.Timestamp) -> float:
        """Third Friday of Mar/Jun/Sep/Dec → vol spike."""
        if dt.month not in (3, 6, 9, 12):
            return 0.0
        first = pd.Timestamp(dt.year, dt.month, 1)
        first_fri = first + pd.offsets.Week(weekday=4)
        if first_fri.month != dt.month:
            first_fri += pd.Timedelta(days=7)
        third_fri = first_fri + pd.Timedelta(days=14)
        if abs((dt - third_fri).days) <= 1:
            return -0.5  # bearish bias from vol
        return 0.0

    @staticmethod
    def is_santa_rally(dt: pd.Timestamp) -> float:
        """Last 5 trading days of Dec + first 2 of Jan → bullish."""
        if dt.month == 12 and dt.day >= 24:
            return 1.0
        if dt.month == 1 and dt.day <= 3:
            return 0.8
        return 0.0

    @staticmethod
    def is_sell_in_may(dt: pd.Timestamp) -> float:
        """May-Oct → slight negative; Nov-Apr → slight positive."""
        if 5 <= dt.month <= 10:
            return -0.3
        return 0.3

    @staticmethod
    def is_monday_effect(dt: pd.Timestamp) -> float:
        """Monday → slight negative bias."""
        if dt.dayofweek == 0:  # Monday
            return -0.4
        if dt.dayofweek == 4:  # Friday → slight positive
            return 0.2
        return 0.0

    @staticmethod
    def is_month_end(dt: pd.Timestamp) -> float:
        """Last 3 trading days → positive (rebalancing flows)."""
        days_in_month = pd.Timestamp(dt.year, dt.month, 1).days_in_month
        if dt.day >= days_in_month - 3:
            return 0.6
        return 0.0

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def score_day(self, dt: pd.Timestamp) -> CalendarScore:
        """Compute all effect scores for a single day."""
        tom = self.is_turn_of_month(dt)
        opex = self.is_opex_week(dt)
        fomc = self.is_fomc_drift(dt)
        quad = self.is_quad_witching(dt)
        santa = self.is_santa_rally(dt)
        sell_may = self.is_sell_in_may(dt)
        monday = self.is_monday_effect(dt)
        month_end = self.is_month_end(dt)

        w = self.weights
        composite = (
            w.get("tom", 0) * tom
            + w.get("opex", 0) * opex
            + w.get("fomc", 0) * fomc
            + w.get("quad_witch", 0) * quad
            + w.get("santa", 0) * santa
            + w.get("sell_may", 0) * sell_may
            + w.get("monday", 0) * monday
            + w.get("month_end", 0) * month_end
        )
        composite = max(-1.0, min(1.0, composite))

        return CalendarScore(
            date=dt, tom=tom, opex=opex, fomc=fomc,
            quad_witch=quad, santa=santa, sell_may=sell_may,
            monday=monday, month_end=month_end, composite=composite,
        )

    def score_series(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute scores for a full date series."""
        rows = []
        for dt in dates:
            s = self.score_day(dt)
            rows.append({
                "date": s.date, "tom": s.tom, "opex": s.opex,
                "fomc": s.fomc, "quad_witch": s.quad_witch,
                "santa": s.santa, "sell_may": s.sell_may,
                "monday": s.monday, "month_end": s.month_end,
                "composite": s.composite,
            })
        return pd.DataFrame(rows).set_index("date")

    def signal_series(self, dates: pd.DatetimeIndex, threshold: float = 0.15) -> pd.Series:
        """Generate +1/-1/0 signal from composite score."""
        df = self.score_series(dates)
        sig = df["composite"].apply(
            lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))
        return sig

    # ------------------------------------------------------------------
    # Effect significance testing
    # ------------------------------------------------------------------

    def test_effects(
        self, dates: pd.DatetimeIndex, returns: pd.Series,
    ) -> List[EffectStats]:
        """Statistical test of each calendar effect."""
        df = self.score_series(dates)
        aligned = pd.DataFrame({"ret": returns}).join(df).dropna()
        baseline = float(aligned["ret"].mean())

        effects = ["tom", "opex", "fomc", "quad_witch", "santa",
                    "sell_may", "monday", "month_end"]
        results: List[EffectStats] = []

        for eff in effects:
            active = aligned[aligned[eff] != 0]["ret"]
            inactive = aligned[aligned[eff] == 0]["ret"]
            n = len(active)
            if n < 5:
                results.append(EffectStats(eff, n, 0, baseline, 0, 0, 1.0, False))
                continue
            avg = float(active.mean())
            excess = avg - baseline
            # Welch t-test
            s1 = float(active.std())
            s2 = float(inactive.std()) if len(inactive) > 1 else s1
            n2 = len(inactive)
            se = math.sqrt(s1**2 / max(n, 1) + s2**2 / max(n2, 1))
            t_stat = excess / se if se > 1e-12 else 0
            # Approximate p-value (two-sided)
            from scipy import stats as sp_stats
            p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(n - 1, 1))))

            results.append(EffectStats(
                eff, n, avg, baseline, excess, t_stat, p_val, p_val < 0.10))

        return results

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        dates: pd.DatetimeIndex,
        returns: pd.Series,
        threshold: float = 0.15,
        cost: float = 0.0005,
    ) -> CalendarBacktestResult:
        """Backtest calendar timing as standalone strategy."""
        sig = self.signal_series(dates, threshold)
        aligned = pd.DataFrame({"sig": sig, "ret": returns}).dropna()

        if len(aligned) < 20:
            return CalendarBacktestResult(0, 0, 0, 0, 0, 0, 0, [])

        pos = aligned["sig"].shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        strat_ret = pos * aligned["ret"] - trades * cost

        r = strat_ret
        total = float((1 + r).prod() - 1)
        n_years = len(r) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1
        mu = float(r.mean())
        std = float(r.std())
        sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
        eq = (1 + r).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())

        n_active = int((pos != 0).sum())
        effects = self.test_effects(dates, returns)

        return CalendarBacktestResult(
            total_return=total, annual_return=annual,
            sharpe=sharpe, max_drawdown=dd,
            n_active_days=n_active, n_total_days=len(aligned),
            active_pct=n_active / len(aligned) if len(aligned) > 0 else 0,
            effect_stats=effects,
        )

    # ------------------------------------------------------------------
    # Overlay: filter EXP-880 signals by calendar score
    # ------------------------------------------------------------------

    def overlay_filter(
        self,
        base_signal: pd.Series,
        dates: pd.DatetimeIndex,
        block_threshold: float = -0.2,
    ) -> pd.Series:
        """Block base strategy trades when calendar score is very negative.

        Returns modified signal with blocked days zeroed out.
        """
        scores = self.score_series(dates)
        composite = scores["composite"].reindex(base_signal.index).fillna(0)
        filtered = base_signal.copy()
        filtered[composite < block_threshold] = 0
        return filtered

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: CalendarBacktestResult,
        scores_df: Optional[pd.DataFrame] = None,
        output_path: str = "reports/calendar_effects.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Effect significance table
        eff_rows = []
        for e in result.effect_stats:
            color = "#059669" if e.significant else "#64748b"
            eff_rows.append(
                f"<tr><td style='text-align:left'>{e.name}</td>"
                f"<td>{e.n_days}</td>"
                f"<td>{e.avg_return * 10000:+.1f}</td>"
                f"<td>{e.excess_return * 10000:+.1f}</td>"
                f"<td>{e.t_stat:.2f}</td>"
                f"<td>{e.p_value:.3f}</td>"
                f"<td style='color:{color};font-weight:700'>"
                f"{'SIG' if e.significant else 'ns'}</td></tr>")

        # Score timeline SVG
        score_svg = ""
        if scores_df is not None and len(scores_df) > 10:
            vals = scores_df["composite"].values
            n = len(vals)
            w, h = 750, 180
            pad = 50
            pw, ph = w - 2 * pad, h - 55
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 30 + (1 - (v + 1) / 2) * ph  # -1..+1 → top..bottom

            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Daily Calendar Score</text>')
            # Zero line
            zy = ty(0)
            parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" '
                          f'stroke="#cbd5e1" stroke-dasharray="3,3"/>')
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(float(vals[i])):.1f}"
                          for i in range(n))
            parts.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
            parts.append("</svg>")
            score_svg = "\n".join(parts)

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Calendar Effects</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
</style></head><body>
<h1>EXP-1150-max: Calendar Effects Alpha</h1>
<div class="card">
<p><strong>Annual Return:</strong> {r.annual_return:.1%} |
<strong>Sharpe:</strong> {r.sharpe:.2f} |
<strong>Max DD:</strong> {r.max_drawdown:.1%} |
<strong>Active Days:</strong> {r.active_pct:.0%}</p>
</div>

{score_svg}

<h2>Effect Significance (p &lt; 0.10)</h2>
<table>
<tr><th style='text-align:left'>Effect</th><th>Days</th><th>Avg (bps)</th>
<th>Excess (bps)</th><th>t-stat</th><th>p-value</th><th>Sig?</th></tr>
{''.join(eff_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
