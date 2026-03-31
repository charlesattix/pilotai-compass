"""Intraday pattern analyzer – identifies time-of-day edges, day-of-week
patterns, opening vs closing execution quality, and intraday volatility
profiles for credit spread strategies.

Provides:
  1. Hour-of-day P&L and win-rate analysis
  2. Day-of-week performance patterns
  3. Opening vs closing execution quality comparison
  4. Intraday volatility curve (hourly vol profile)
  5. Optimal trade timing recommendations per regime
  6. Pre/post-market analysis
  7. HTML report with heatmap, bar charts, and tables
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 16
DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday"]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class HourStats:
    """Performance statistics for one hour of the day."""
    hour: int
    avg_pnl: float
    total_pnl: float
    win_rate: float
    n_trades: int
    avg_volatility: float
    session: str          # "pre_market", "market", "post_market"


@dataclass
class DayOfWeekStats:
    """Performance statistics for one day of the week."""
    day: str              # Monday … Friday
    day_idx: int          # 0=Mon … 6=Sun
    avg_pnl: float
    total_pnl: float
    win_rate: float
    n_trades: int
    avg_volatility: float


@dataclass
class SessionComparison:
    """Opening vs closing execution quality."""
    open_avg_pnl: float
    open_win_rate: float
    open_n_trades: int
    close_avg_pnl: float
    close_win_rate: float
    close_n_trades: int
    open_avg_slippage: float
    close_avg_slippage: float
    better_session: str   # "open" or "close"


@dataclass
class TimingRecommendation:
    """Optimal trade timing for a regime."""
    regime: str
    best_entry_hour: int
    best_exit_hour: int
    best_day: str
    worst_day: str
    avg_pnl_at_best: float
    n_obs: int


@dataclass
class VolatilityProfile:
    """Intraday volatility curve."""
    hourly_vol: Dict[int, float]   # hour → annualised vol
    peak_hour: int
    trough_hour: int
    open_close_ratio: float        # vol at open / vol at close


@dataclass
class IntradayResult:
    """Complete intraday analysis output."""
    hour_stats: List[HourStats] = field(default_factory=list)
    dow_stats: List[DayOfWeekStats] = field(default_factory=list)
    session_comparison: Optional[SessionComparison] = None
    timing_recommendations: List[TimingRecommendation] = field(default_factory=list)
    volatility_profile: Optional[VolatilityProfile] = None
    n_trades: int = 0
    generated_at: str = ""


# ── Core analyzer ───────────────────────────────────────────────────────────
class IntradayPatternAnalyzer:
    """Identifies intraday trading patterns and optimal timing."""

    def __init__(
        self,
        market_open: int = MARKET_OPEN_HOUR,
        market_close: int = MARKET_CLOSE_HOUR,
    ) -> None:
        self.market_open = market_open
        self.market_close = market_close

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        trades: pd.DataFrame,
        price_series: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> IntradayResult:
        """Run full intraday analysis.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have columns: datetime (or date+hour), pnl.
            Optional: hour, day_of_week, slippage, regime, session.
        price_series : pd.Series, optional
            Intraday price series for volatility profiling.
        regimes : pd.Series, optional
            Regime labels indexed by datetime.
        """
        trades = self._normalize(trades)
        if trades.empty:
            return IntradayResult(generated_at=self._now())

        hour_stats = self._hour_analysis(trades)
        dow_stats = self._dow_analysis(trades)
        session = self._session_comparison(trades)

        vol_profile: Optional[VolatilityProfile] = None
        if price_series is not None and len(price_series) > 20:
            vol_profile = self._volatility_profile(price_series)

        timing: List[TimingRecommendation] = []
        if regimes is not None:
            trades_with_regime = self._merge_regimes(trades, regimes)
            timing = self._timing_recommendations(trades_with_regime)
        elif "regime" in trades.columns:
            timing = self._timing_recommendations(trades)

        return IntradayResult(
            hour_stats=hour_stats,
            dow_stats=dow_stats,
            session_comparison=session,
            timing_recommendations=timing,
            volatility_profile=vol_profile,
            n_trades=len(trades),
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: IntradayResult,
        output_path: str | Path = "reports/intraday_patterns.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Intraday patterns report written to %s", path)
        return path

    # ── Normalization ───────────────────────────────────────────────────────
    def _normalize(self, trades: pd.DataFrame) -> pd.DataFrame:
        df = trades.copy()
        if "pnl" not in df.columns:
            return pd.DataFrame()

        # Ensure datetime column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        elif "entry_date" in df.columns:
            df["datetime"] = pd.to_datetime(df["entry_date"])
        else:
            return pd.DataFrame()

        # Extract hour and dow if not present
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["datetime"].dt.dayofweek

        # Session classification
        if "session" not in df.columns:
            df["session"] = df["hour"].apply(self._classify_session)

        return df

    def _classify_session(self, hour: int) -> str:
        if hour < self.market_open:
            return "pre_market"
        if hour >= self.market_close:
            return "post_market"
        return "market"

    # ── Hour-of-day analysis ────────────────────────────────────────────────
    def _hour_analysis(self, trades: pd.DataFrame) -> List[HourStats]:
        results: List[HourStats] = []
        for hour in range(24):
            mask = trades["hour"] == hour
            subset = trades[mask]
            if subset.empty:
                continue
            pnls = subset["pnl"]
            results.append(HourStats(
                hour=hour,
                avg_pnl=float(pnls.mean()),
                total_pnl=float(pnls.sum()),
                win_rate=float((pnls > 0).mean()),
                n_trades=len(subset),
                avg_volatility=float(pnls.std()) if len(pnls) > 1 else 0.0,
                session=self._classify_session(hour),
            ))
        return results

    # ── Day-of-week analysis ────────────────────────────────────────────────
    @staticmethod
    def _dow_analysis(trades: pd.DataFrame) -> List[DayOfWeekStats]:
        results: List[DayOfWeekStats] = []
        for dow in range(7):
            mask = trades["day_of_week"] == dow
            subset = trades[mask]
            if subset.empty:
                continue
            pnls = subset["pnl"]
            results.append(DayOfWeekStats(
                day=DOW_LABELS[dow],
                day_idx=dow,
                avg_pnl=float(pnls.mean()),
                total_pnl=float(pnls.sum()),
                win_rate=float((pnls > 0).mean()),
                n_trades=len(subset),
                avg_volatility=float(pnls.std()) if len(pnls) > 1 else 0.0,
            ))
        return results

    # ── Session comparison ──────────────────────────────────────────────────
    def _session_comparison(self, trades: pd.DataFrame) -> SessionComparison:
        first_hour = trades["hour"].min()
        last_hour = trades["hour"].max()

        open_mask = trades["hour"] <= first_hour + 1
        close_mask = trades["hour"] >= last_hour - 1

        open_t = trades[open_mask]
        close_t = trades[close_mask]

        open_slip = float(open_t["slippage"].mean()) if "slippage" in open_t.columns and not open_t.empty else 0.0
        close_slip = float(close_t["slippage"].mean()) if "slippage" in close_t.columns and not close_t.empty else 0.0

        open_pnl = float(open_t["pnl"].mean()) if not open_t.empty else 0.0
        close_pnl = float(close_t["pnl"].mean()) if not close_t.empty else 0.0
        open_wr = float((open_t["pnl"] > 0).mean()) if not open_t.empty else 0.0
        close_wr = float((close_t["pnl"] > 0).mean()) if not close_t.empty else 0.0

        return SessionComparison(
            open_avg_pnl=open_pnl,
            open_win_rate=open_wr,
            open_n_trades=len(open_t),
            close_avg_pnl=close_pnl,
            close_win_rate=close_wr,
            close_n_trades=len(close_t),
            open_avg_slippage=open_slip,
            close_avg_slippage=close_slip,
            better_session="open" if open_pnl >= close_pnl else "close",
        )

    # ── Volatility profile ──────────────────────────────────────────────────
    def _volatility_profile(self, prices: pd.Series) -> VolatilityProfile:
        prices = prices.dropna()
        idx = prices.index
        if not hasattr(idx, "hour"):
            prices.index = pd.to_datetime(prices.index)
            idx = prices.index

        returns = prices.pct_change().dropna()
        hourly_vol: Dict[int, float] = {}
        for hour in range(24):
            mask = returns.index.hour == hour
            hr = returns[mask]
            if len(hr) > 1:
                hourly_vol[hour] = float(hr.std() * np.sqrt(252 * 6.5))
            else:
                hourly_vol[hour] = 0.0

        active = {h: v for h, v in hourly_vol.items() if v > 0}
        peak = max(active, key=active.get) if active else 0
        trough = min(active, key=active.get) if active else 0

        open_vol = hourly_vol.get(self.market_open, 0.0)
        close_vol = hourly_vol.get(self.market_close - 1, 0.0)
        ratio = open_vol / close_vol if close_vol > 1e-12 else 0.0

        return VolatilityProfile(
            hourly_vol=hourly_vol,
            peak_hour=peak,
            trough_hour=trough,
            open_close_ratio=ratio,
        )

    # ── Regime-specific timing ──────────────────────────────────────────────
    @staticmethod
    def _merge_regimes(trades: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        df = trades.copy()
        regimes = regimes.reindex(df["datetime"], method="nearest")
        df["regime"] = regimes.values
        return df

    @staticmethod
    def _timing_recommendations(trades: pd.DataFrame) -> List[TimingRecommendation]:
        if "regime" not in trades.columns:
            return []
        results: List[TimingRecommendation] = []
        for regime in sorted(trades["regime"].dropna().unique()):
            sub = trades[trades["regime"] == regime]
            if len(sub) < 3:
                continue

            # Best entry hour by avg pnl
            hour_pnl = sub.groupby("hour")["pnl"].mean()
            best_h = int(hour_pnl.idxmax()) if not hour_pnl.empty else 0
            worst_h = int(hour_pnl.idxmin()) if not hour_pnl.empty else 0

            # Best / worst day
            dow_pnl = sub.groupby("day_of_week")["pnl"].mean()
            best_d = int(dow_pnl.idxmax()) if not dow_pnl.empty else 0
            worst_d = int(dow_pnl.idxmin()) if not dow_pnl.empty else 0

            results.append(TimingRecommendation(
                regime=str(regime),
                best_entry_hour=best_h,
                best_exit_hour=worst_h,
                best_day=DOW_LABELS[best_d] if best_d < 7 else str(best_d),
                worst_day=DOW_LABELS[worst_d] if worst_d < 7 else str(worst_d),
                avg_pnl_at_best=float(hour_pnl.loc[best_h]) if best_h in hour_pnl.index else 0.0,
                n_obs=len(sub),
            ))
        return results

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: IntradayResult) -> str:
        cards = self._html_cards(r)
        heatmap = self._svg_hour_heatmap(r.hour_stats)
        dow_bars = self._svg_dow_bars(r.dow_stats)
        session_tbl = self._html_session(r.session_comparison)
        timing_tbl = self._html_timing(r.timing_recommendations)
        vol_tbl = self._html_volatility(r.volatility_profile)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Intraday Pattern Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
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
</style>
</head>
<body>
<h1>Intraday Pattern Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_trades} trades analyzed</p>

{cards}

<div class="sec">
<h2>Hour-of-Day Performance Heatmap</h2>
{heatmap}
</div>

<div class="sec">
<h2>Day-of-Week Performance</h2>
{dow_bars}
</div>

{session_tbl}
{timing_tbl}
{vol_tbl}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: IntradayResult) -> str:
        best_h = max(r.hour_stats, key=lambda h: h.avg_pnl) if r.hour_stats else None
        worst_h = min(r.hour_stats, key=lambda h: h.avg_pnl) if r.hour_stats else None
        best_d = max(r.dow_stats, key=lambda d: d.avg_pnl) if r.dow_stats else None
        sess = r.session_comparison
        return f"""<div class="grid">
<div class="card"><div class="lbl">Best Hour</div><div class="val">{best_h.hour:02d}:00</div></div>
<div class="card"><div class="lbl">Worst Hour</div><div class="val">{worst_h.hour:02d}:00</div></div>
<div class="card"><div class="lbl">Best Day</div><div class="val">{best_d.day if best_d else 'N/A'}</div></div>
<div class="card"><div class="lbl">Better Session</div><div class="val">{sess.better_session.upper() if sess else 'N/A'}</div></div>
<div class="card"><div class="lbl">Trades</div><div class="val">{r.n_trades}</div></div>
</div>""" if r.hour_stats else ""

    @staticmethod
    def _svg_hour_heatmap(hours: List[HourStats]) -> str:
        if not hours:
            return "<p>No hourly data.</p>"
        w, h = 600, 80
        pl = 30
        cell_w = (w - pl) / 24
        cells = ""
        max_abs = max(abs(s.avg_pnl) for s in hours) or 1.0
        hour_map = {s.hour: s for s in hours}

        for hr in range(24):
            x = pl + hr * cell_w
            s = hour_map.get(hr)
            if s:
                intensity = s.avg_pnl / max_abs
                if intensity >= 0:
                    r_c, g_c, b_c = 30, int(100 + 155 * intensity), 50
                else:
                    r_c, g_c, b_c = int(100 + 155 * abs(intensity)), 30, 30
                colour = f"rgb({r_c},{g_c},{b_c})"
            else:
                colour = "#1e293b"
            cells += (
                f'<rect x="{x:.0f}" y="10" width="{cell_w:.0f}" height="40" '
                f'fill="{colour}" stroke="#0f172a" stroke-width="1" rx="2"/>'
                f'<text x="{x + cell_w / 2:.0f}" y="65" text-anchor="middle" '
                f'font-size="9" fill="#94a3b8">{hr}</text>'
            )

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'{cells}</svg>'
        )

    @staticmethod
    def _svg_dow_bars(days: List[DayOfWeekStats]) -> str:
        if not days:
            return "<p>No day-of-week data.</p>"
        w, h = 450, 180
        pl, pb, pt = 70, 35, 15
        ch = h - pb - pt
        mid_y = pt + ch // 2
        max_abs = max(abs(d.avg_pnl) for d in days) or 1.0
        n = len(days)
        bar_w = min(45, (w - pl) // n - 8)

        bars = ""
        for i, d in enumerate(days):
            x = pl + i * ((w - pl) // n) + 4
            scaled = (d.avg_pnl / max_abs) * (ch * 0.4)
            if d.avg_pnl >= 0:
                bh = scaled
                y = mid_y - bh
                colour = "#4ade80"
            else:
                bh = -scaled
                y = mid_y
                colour = "#f87171"
            bars += (
                f'<rect x="{x}" y="{y:.0f}" width="{bar_w}" height="{bh:.0f}" '
                f'rx="3" fill="{colour}" opacity="0.85"/>'
                f'<text x="{x + bar_w // 2}" y="{y - 4:.0f}" text-anchor="middle" '
                f'font-size="10" fill="#e2e8f0">${d.avg_pnl:.0f}</text>'
                f'<text x="{x + bar_w // 2}" y="{h - 8}" text-anchor="middle" '
                f'font-size="10" fill="#94a3b8">{d.day[:3]}</text>'
            )

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{mid_y}" x2="{w}" y2="{mid_y}" '
            f'stroke="#475569" stroke-width="1" stroke-dasharray="4"/>'
            f'{bars}</svg>'
        )

    @staticmethod
    def _html_session(sess: Optional[SessionComparison]) -> str:
        if not sess:
            return ""
        o_cls = "pos" if sess.open_avg_pnl >= 0 else "neg"
        c_cls = "pos" if sess.close_avg_pnl >= 0 else "neg"
        return f"""<div class="sec">
<h2>Opening vs Closing Execution</h2>
<table>
<thead><tr><th>Metric</th><th>Open</th><th>Close</th></tr></thead>
<tbody>
<tr><td>Avg P&L</td><td class="{o_cls}">${sess.open_avg_pnl:.2f}</td><td class="{c_cls}">${sess.close_avg_pnl:.2f}</td></tr>
<tr><td>Win Rate</td><td>{sess.open_win_rate:.1%}</td><td>{sess.close_win_rate:.1%}</td></tr>
<tr><td>Trades</td><td>{sess.open_n_trades}</td><td>{sess.close_n_trades}</td></tr>
<tr><td>Avg Slippage</td><td>{sess.open_avg_slippage:.4f}</td><td>{sess.close_avg_slippage:.4f}</td></tr>
<tr><td>Better?</td><td colspan="2" class="pos"><strong>{sess.better_session.upper()}</strong></td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_timing(recs: List[TimingRecommendation]) -> str:
        if not recs:
            return ""
        rows = ""
        for t in recs:
            rows += (
                f"<tr><td>{t.regime}</td>"
                f"<td>{t.best_entry_hour:02d}:00</td>"
                f"<td>{t.best_exit_hour:02d}:00</td>"
                f"<td>{t.best_day}</td>"
                f"<td>{t.worst_day}</td>"
                f"<td>${t.avg_pnl_at_best:.2f}</td>"
                f"<td>{t.n_obs}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Regime-Specific Timing Recommendations</h2>
<table>
<thead><tr><th>Regime</th><th>Best Entry</th><th>Best Exit</th><th>Best Day</th><th>Worst Day</th><th>Avg P&L</th><th>Obs</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_volatility(vp: Optional[VolatilityProfile]) -> str:
        if not vp:
            return ""
        rows = ""
        for h in sorted(vp.hourly_vol):
            v = vp.hourly_vol[h]
            if v > 0:
                rows += f"<tr><td>{h:02d}:00</td><td>{v:.2%}</td></tr>"
        return f"""<div class="sec">
<h2>Intraday Volatility Profile</h2>
<table>
<thead><tr><th>Hour</th><th>Annualised Vol</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p style="color:#94a3b8;font-size:.8rem;margin-top:8px">
Peak: {vp.peak_hour:02d}:00 &middot; Trough: {vp.trough_hour:02d}:00 &middot;
Open/Close ratio: {vp.open_close_ratio:.2f}x
</p>
</div>"""
