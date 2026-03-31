"""
Trade journal and analytics — post-trade analysis and reporting.

Provides structured analysis of closed trades from backtest or paper trading:
  1. Model-based P&L attribution (theta, delta, vega, residual)
  2. Win/loss streak analysis with regime context
  3. Day-of-week and time-based edge analysis
  4. Monthly/quarterly performance rollups
  5. Self-contained HTML report

P&L Attribution Model
---------------------
Since raw Greeks are not available in the training data, we decompose P&L
using a heuristic model based on observable trade characteristics:

  **Theta component**: Credit spreads collect premium via time decay.
    theta_pnl ≈ net_credit × (hold_days / dte_at_entry) × contracts × 100
    Capped at the actual P&L for profitable trades to prevent overattribution.

  **Delta component**: Directional moves affect spread value.
    delta_proxy ≈ pnl_sign × |spy_price_change_pct| × spread_exposure
    Where spread_exposure = contracts × spread_width × 100

  **Vega component**: Volatility changes (VIX at entry as proxy).
    vega_pnl ≈ -realized_vol_change × vega_sensitivity
    Higher VIX at entry → more premium collected → bigger vega contribution.

  **Residual**: pnl - theta - delta - vega (captures gamma, execution, etc.)

Usage::

    from compass.trade_journal import TradeJournal
    journal = TradeJournal.from_csv("compass/training_data_combined.csv")
    html = journal.generate_report("reports/trade_journal.html")
"""

from __future__ import annotations

import base64
import io
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "trade_journal.html"


# ── P&L Attribution ──────────────────────────────────────────────────────


def attribute_pnl(trade: pd.Series) -> Dict[str, float]:
    """Decompose a single trade's P&L into theta, delta, vega, residual.

    All components sum to the actual P&L (no leakage).
    Returns dict with keys: theta, delta, vega, residual, total.
    """
    pnl = trade.get("pnl", 0.0)
    if pd.isna(pnl):
        return {"theta": 0, "delta": 0, "vega": 0, "residual": 0, "total": 0}

    credit = abs(trade.get("net_credit", 0) or 0)
    dte = trade.get("dte_at_entry", 30) or 30
    hold = trade.get("hold_days", 1) or 1
    contracts = trade.get("contracts", 1) or 1
    spread_w = abs(trade.get("spread_width", 5) or 5)
    vix = trade.get("vix", 20) or 20
    rv_20d = trade.get("realized_vol_20d", 20) or 20
    momentum = trade.get("momentum_10d_pct", 0) or 0

    # Theta: time decay of credit collected
    # For credit spreads, max theta capture = full credit × contracts × multiplier
    max_theta = credit * contracts * 100
    decay_fraction = min(1.0, hold / max(dte, 1))
    theta = max_theta * decay_fraction

    # For losing trades, theta was still earned but overwhelmed by other Greeks
    # Cap theta at max possible gain
    if pnl > 0:
        theta = min(theta, pnl)
    else:
        theta = min(theta, max_theta * 0.5)  # still earned some theta

    # Delta: directional P&L from underlying move
    # Use momentum as proxy for price change during the trade
    spread_notional = contracts * spread_w * 100
    delta_sensitivity = 0.3 if spread_w > 0 else 0.0  # typical spread delta
    delta = momentum / 100.0 * spread_notional * delta_sensitivity

    # For bull puts, positive momentum helps; for bear calls, negative helps
    spread_type = str(trade.get("spread_type", ""))
    if "bear" in spread_type:
        delta = -delta

    # Vega: vol change impact
    # Higher VIX at entry = more premium → if vol drops, we profit from vega
    vol_premium = max(0, (vix - rv_20d)) / 100.0  # IV-RV spread as proxy
    vega = -vol_premium * contracts * spread_w * 100 * 0.1  # vega sensitivity factor

    # Residual: whatever is left (gamma, execution, model error)
    residual = pnl - theta - delta - vega

    return {
        "theta": round(theta, 2),
        "delta": round(delta, 2),
        "vega": round(vega, 2),
        "residual": round(residual, 2),
        "total": round(pnl, 2),
    }


def attribute_all(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute P&L attribution for all trades. Returns DataFrame with Greek columns."""
    attrs = trades.apply(attribute_pnl, axis=1, result_type="expand")
    return pd.concat([trades, attrs.add_prefix("attr_")], axis=1)


# ── Streak Analysis ──────────────────────────────────────────────────────


@dataclass
class Streak:
    """A consecutive win or loss streak."""
    streak_type: str      # "win" or "loss"
    length: int
    total_pnl: float
    start_date: str
    end_date: str
    regimes: List[str]
    avg_pnl: float


def compute_streaks(trades: pd.DataFrame) -> List[Streak]:
    """Identify consecutive win/loss streaks from chronologically sorted trades."""
    if len(trades) == 0 or "win" not in trades.columns:
        return []

    streaks: List[Streak] = []
    current_type = None
    current_trades: List[pd.Series] = []

    for _, trade in trades.iterrows():
        w = trade.get("win", 0)
        trade_type = "win" if w == 1 else "loss"

        if trade_type != current_type:
            if current_trades:
                streaks.append(_build_streak(current_type, current_trades))
            current_type = trade_type
            current_trades = [trade]
        else:
            current_trades.append(trade)

    if current_trades:
        streaks.append(_build_streak(current_type, current_trades))

    return streaks


def _build_streak(streak_type: str, trades: List[pd.Series]) -> Streak:
    pnls = [t.get("pnl", 0) or 0 for t in trades]
    regimes = [str(t.get("regime", "?")) for t in trades]
    total = sum(pnls)
    return Streak(
        streak_type=streak_type,
        length=len(trades),
        total_pnl=round(total, 2),
        start_date=str(trades[0].get("entry_date", "?")),
        end_date=str(trades[-1].get("exit_date", "?")),
        regimes=regimes,
        avg_pnl=round(total / len(trades), 2) if trades else 0,
    )


def streak_summary(streaks: List[Streak]) -> Dict[str, Any]:
    """Aggregate streak statistics."""
    if not streaks:
        return {"max_win_streak": 0, "max_loss_streak": 0,
                "avg_win_streak": 0, "avg_loss_streak": 0}

    wins = [s for s in streaks if s.streak_type == "win"]
    losses = [s for s in streaks if s.streak_type == "loss"]

    return {
        "max_win_streak": max((s.length for s in wins), default=0),
        "max_loss_streak": max((s.length for s in losses), default=0),
        "avg_win_streak": round(np.mean([s.length for s in wins]), 1) if wins else 0,
        "avg_loss_streak": round(np.mean([s.length for s in losses]), 1) if losses else 0,
        "total_streaks": len(streaks),
        "longest_streak": max(streaks, key=lambda s: s.length) if streaks else None,
    }


# ── Day-of-Week / Time Edge Analysis ────────────────────────────────────


def day_of_week_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate, avg P&L, and count by day of week (entry day)."""
    if "day_of_week" not in trades.columns or len(trades) == 0:
        return pd.DataFrame()

    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    df = trades.copy()
    df["day_name"] = df["day_of_week"].map(dow_map)

    grouped = df.groupby("day_name", observed=True)
    result = pd.DataFrame({
        "day": grouped.size().index,
        "count": grouped.size().values,
        "win_rate": grouped["win"].mean().values if "win" in df.columns else None,
        "avg_pnl": grouped["pnl"].mean().values if "pnl" in df.columns else None,
        "total_pnl": grouped["pnl"].sum().values if "pnl" in df.columns else None,
        "avg_return_pct": grouped["return_pct"].mean().values if "return_pct" in df.columns else None,
    })

    # Reorder Mon-Fri
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    result["_order"] = result["day"].map({d: i for i, d in enumerate(day_order)})
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return result


def regime_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Performance breakdown by market regime."""
    if "regime" not in trades.columns or len(trades) == 0:
        return pd.DataFrame()

    grouped = trades.groupby("regime", observed=True)
    result = pd.DataFrame({
        "regime": grouped.size().index,
        "count": grouped.size().values,
        "win_rate": grouped["win"].mean().values if "win" in trades.columns else None,
        "avg_pnl": grouped["pnl"].mean().values if "pnl" in trades.columns else None,
        "total_pnl": grouped["pnl"].sum().values if "pnl" in trades.columns else None,
    })
    return result.sort_values("count", ascending=False).reset_index(drop=True)


# ── Monthly/Quarterly Rollups ────────────────────────────────────────────


def monthly_rollup(trades: pd.DataFrame) -> pd.DataFrame:
    """Monthly performance summary."""
    if len(trades) == 0:
        return pd.DataFrame()

    df = trades.copy()
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["month"] = df["exit_dt"].dt.to_period("M")

    grouped = df.groupby("month")
    result = pd.DataFrame({
        "month": [str(m) for m in grouped.size().index],
        "trades": grouped.size().values,
        "wins": grouped["win"].sum().values if "win" in df.columns else 0,
        "win_rate": grouped["win"].mean().values if "win" in df.columns else None,
        "total_pnl": grouped["pnl"].sum().values if "pnl" in df.columns else None,
        "avg_pnl": grouped["pnl"].mean().values if "pnl" in df.columns else None,
        "best_trade": grouped["pnl"].max().values if "pnl" in df.columns else None,
        "worst_trade": grouped["pnl"].min().values if "pnl" in df.columns else None,
    })
    return result


def quarterly_rollup(trades: pd.DataFrame) -> pd.DataFrame:
    """Quarterly performance summary."""
    if len(trades) == 0:
        return pd.DataFrame()

    df = trades.copy()
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["quarter"] = df["exit_dt"].dt.to_period("Q")

    grouped = df.groupby("quarter")
    result = pd.DataFrame({
        "quarter": [str(q) for q in grouped.size().index],
        "trades": grouped.size().values,
        "win_rate": grouped["win"].mean().values if "win" in df.columns else None,
        "total_pnl": grouped["pnl"].sum().values if "pnl" in df.columns else None,
        "avg_pnl": grouped["pnl"].mean().values if "pnl" in df.columns else None,
    })
    return result


# ── TradeJournal Class ───────────────────────────────────────────────────


class TradeJournal:
    """Structured trade journal with analytics and reporting.

    Loads trades, computes attribution, streaks, and dimensional analysis,
    then generates an HTML report.

    Args:
        trades: DataFrame of closed trades (must have pnl, entry_date, exit_date).
        starting_capital: For return calculations.
    """

    def __init__(self, trades: pd.DataFrame, starting_capital: float = 100_000):
        self.raw_trades = trades.sort_values("entry_date").reset_index(drop=True)
        self.starting_capital = starting_capital
        self.trades = attribute_all(self.raw_trades)
        self.streaks = compute_streaks(self.raw_trades)

    @classmethod
    def from_csv(cls, csv_path: str, starting_capital: float = 100_000) -> "TradeJournal":
        df = pd.read_csv(csv_path)
        return cls(df, starting_capital)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return float(self.trades["pnl"].sum())

    @property
    def win_rate(self) -> Optional[float]:
        if "win" not in self.trades.columns or len(self.trades) == 0:
            return None
        return float(self.trades["win"].mean())

    def attribution_summary(self) -> Dict[str, float]:
        """Aggregate P&L attribution across all trades."""
        cols = ["attr_theta", "attr_delta", "attr_vega", "attr_residual", "attr_total"]
        present = [c for c in cols if c in self.trades.columns]
        if not present:
            return {}
        totals = self.trades[present].sum()
        return {k.replace("attr_", ""): round(v, 2) for k, v in totals.items()}

    def day_of_week(self) -> pd.DataFrame:
        return day_of_week_analysis(self.trades)

    def regime_breakdown(self) -> pd.DataFrame:
        return regime_analysis(self.trades)

    def monthly(self) -> pd.DataFrame:
        return monthly_rollup(self.trades)

    def quarterly(self) -> pd.DataFrame:
        return quarterly_rollup(self.trades)

    def streak_stats(self) -> Dict[str, Any]:
        return streak_summary(self.streaks)

    # ── HTML Report ──────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate self-contained HTML trade journal report."""
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Trade journal written to %s (%d bytes)", out, len(html))
        return str(out.resolve())

    def _render_charts(self) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        charts["attribution"] = self._chart_attribution()
        charts["monthly_pnl"] = self._chart_monthly_pnl()
        charts["cumulative"] = self._chart_cumulative_pnl()
        charts["dow"] = self._chart_day_of_week()
        return charts

    def _fig_to_b64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_attribution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = self.attribution_summary()
        if not summary or "total" not in summary:
            return ""
        components = {k: v for k, v in summary.items() if k != "total"}
        if not components:
            return ""

        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(components.keys())
        vals = list(components.values())
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
        ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_ylabel("P&L ($)")
        ax.set_title("P&L Attribution (aggregate)", fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + (abs(v) * 0.03 if v >= 0 else -abs(v) * 0.08),
                    f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_monthly_pnl(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mo = self.monthly()
        if mo.empty or "total_pnl" not in mo.columns:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in mo["total_pnl"]]
        ax.bar(range(len(mo)), mo["total_pnl"], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(mo)))
        ax.set_xticklabels(mo["month"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("P&L ($)")
        ax.set_title("Monthly P&L", fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_cumulative_pnl(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "pnl" not in self.trades.columns or len(self.trades) < 2:
            return ""

        cumulative = self.trades["pnl"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15, color="#2563eb")
        ax.plot(range(len(cumulative)), cumulative, color="#2563eb", lw=1.5)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)

        # Mark max drawdown point
        hwm = cumulative.cummax()
        dd = cumulative - hwm
        worst_idx = dd.idxmin()
        ax.axvline(worst_idx, color="#dc2626", ls="--", lw=0.8, alpha=0.5, label="Max DD point")

        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("Equity Curve", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_day_of_week(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dow = self.day_of_week()
        if dow.empty:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Win rate
        if dow["win_rate"].notna().any():
            colors = ["#16a34a" if wr >= 0.5 else "#dc2626" for wr in dow["win_rate"]]
            ax1.bar(dow["day"], dow["win_rate"] * 100, color=colors, alpha=0.85)
            ax1.set_ylabel("Win Rate (%)")
            ax1.set_title("Win Rate by Entry Day", fontsize=10)
            ax1.set_ylim(0, 100)
            ax1.axhline(50, color="gray", ls="--", lw=0.8)
            ax1.grid(True, axis="y", alpha=0.3)

        # Avg P&L
        if dow["avg_pnl"].notna().any():
            colors = ["#16a34a" if v >= 0 else "#dc2626" for v in dow["avg_pnl"]]
            ax2.bar(dow["day"], dow["avg_pnl"], color=colors, alpha=0.85)
            ax2.set_ylabel("Avg P&L ($)")
            ax2.set_title("Avg P&L by Entry Day", fontsize=10)
            ax2.axhline(0, color="black", lw=0.5)
            ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        attr = self.attribution_summary()
        ss = self.streak_stats()
        wr = self.win_rate

        # KPI values
        total_pnl = self.total_pnl
        avg_pnl = float(self.trades["pnl"].mean()) if len(self.trades) > 0 else 0
        avg_hold = float(self.trades["hold_days"].mean()) if "hold_days" in self.trades.columns else 0

        # Monthly table rows
        mo = self.monthly()
        monthly_rows = ""
        for _, r in mo.iterrows():
            pnl_cls = "good" if (r.get("total_pnl") or 0) >= 0 else "bad"
            monthly_rows += (
                f'<tr><td>{r["month"]}</td><td>{r["trades"]}</td>'
                f'<td>{r.get("win_rate", 0):.0%}</td>'
                f'<td class="{pnl_cls}">${r.get("total_pnl", 0):,.0f}</td>'
                f'<td>${r.get("avg_pnl", 0):,.0f}</td></tr>\n'
            )

        # Quarterly table rows
        qt = self.quarterly()
        quarterly_rows = ""
        for _, r in qt.iterrows():
            pnl_cls = "good" if (r.get("total_pnl") or 0) >= 0 else "bad"
            quarterly_rows += (
                f'<tr><td>{r["quarter"]}</td><td>{r["trades"]}</td>'
                f'<td>{r.get("win_rate", 0):.0%}</td>'
                f'<td class="{pnl_cls}">${r.get("total_pnl", 0):,.0f}</td></tr>\n'
            )

        # Regime table rows
        reg = self.regime_breakdown()
        regime_rows = ""
        for _, r in reg.iterrows():
            regime_rows += (
                f'<tr><td>{r["regime"]}</td><td>{r["count"]}</td>'
                f'<td>{r.get("win_rate", 0):.0%}</td>'
                f'<td>${r.get("avg_pnl", 0):,.0f}</td>'
                f'<td>${r.get("total_pnl", 0):,.0f}</td></tr>\n'
            )

        # Top streaks
        top_streaks = sorted(self.streaks, key=lambda s: s.length, reverse=True)[:10]
        streak_rows = ""
        for s in top_streaks:
            cls = "good" if s.streak_type == "win" else "bad"
            dominant_regime = Counter(s.regimes).most_common(1)[0][0] if s.regimes else "?"
            streak_rows += (
                f'<tr class="{cls}-row"><td class="{cls}">{s.streak_type.upper()}</td>'
                f'<td>{s.length}</td><td>${s.total_pnl:,.0f}</td>'
                f'<td>{s.start_date} → {s.end_date}</td>'
                f'<td>{dominant_regime}</td></tr>\n'
            )

        def _img(key):
            b64 = charts.get(key, "")
            if b64:
                return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>'
            return '<p class="muted">Insufficient data</p>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trade Journal</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .muted {{ color: #94a3b8; font-style: italic; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .good-row {{ background: #f0fdf4; }}
  .bad-row {{ background: #fef2f2; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Trade Journal</h1>
<div class="meta">{self.n_trades} trades &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{self.n_trades}</div><div class="label">Total Trades</div></div>
  <div class="kpi"><div class="value">{wr:.1%}</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value {'good' if total_pnl >= 0 else 'bad'}">${total_pnl:,.0f}</div><div class="label">Total P&L</div></div>
  <div class="kpi"><div class="value">${avg_pnl:,.0f}</div><div class="label">Avg P&L</div></div>
  <div class="kpi"><div class="value">{avg_hold:.1f}d</div><div class="label">Avg Hold</div></div>
  <div class="kpi"><div class="value">{ss.get('max_win_streak', 0)}</div><div class="label">Max Win Streak</div></div>
  <div class="kpi"><div class="value">{ss.get('max_loss_streak', 0)}</div><div class="label">Max Loss Streak</div></div>
</div>

<h2>1. Equity Curve</h2>
{_img("cumulative")}

<h2>2. P&L Attribution</h2>
<p class="muted">Heuristic decomposition: theta (time decay), delta (directional), vega (vol change), residual (gamma + execution).</p>
{_img("attribution")}
<table style="max-width:400px">
<thead><tr><th>Component</th><th>P&L</th><th>% of Total</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{k.title()}</td><td>${v:,.0f}</td><td>{v/attr.get("total",1)*100:.1f}%</td></tr>' for k, v in attr.items() if k != "total")}
<tr style="border-top:2px solid #cbd5e1"><td><strong>Total</strong></td><td><strong>${attr.get("total",0):,.0f}</strong></td><td>100%</td></tr>
</tbody>
</table>

<h2>3. Monthly P&L</h2>
{_img("monthly_pnl")}
<table>
<thead><tr><th>Month</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Avg P&L</th></tr></thead>
<tbody>{monthly_rows}</tbody>
</table>

<div class="two-col">
<div>
<h2>4. Day-of-Week Edge</h2>
{_img("dow")}
</div>
<div>
<h2>5. Regime Breakdown</h2>
<table>
<thead><tr><th>Regime</th><th>Count</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th></tr></thead>
<tbody>{regime_rows}</tbody>
</table>
</div>
</div>

<h2>6. Quarterly Rollup</h2>
<table>
<thead><tr><th>Quarter</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th></tr></thead>
<tbody>{quarterly_rows}</tbody>
</table>

<h2>7. Top Streaks</h2>
<table>
<thead><tr><th>Type</th><th>Length</th><th>P&L</th><th>Period</th><th>Regime</th></tr></thead>
<tbody>{streak_rows}</tbody>
</table>

<footer>Generated by <code>compass/trade_journal.py</code></footer>
</body>
</html>"""
        return html
