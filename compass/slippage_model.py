"""
Advanced slippage modeling engine.

Models:
  - Fixed bps
  - Volume-dependent (square-root market-impact law)
  - Volatility-adjusted (slippage scales with realised vol)
  - Order-size impact (Almgren-Chriss square-root)

Calibration:
  - Historical slippage calibration from trade data
  - Per-instrument slippage profiles
  - Bid-ask spread cost estimation
  - Time-of-day slippage patterns
  - Slippage budget allocation across portfolio

HTML report at reports/slippage_model.html with model comparison,
calibration fit, instrument/time attribution.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.slippage_model import SlippageEngine
    engine = SlippageEngine()
    result = engine.analyze(trades_df)
    SlippageEngine.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "slippage_model.html"


# ── Slippage model functions ─────────────────────────────────────────────


def slippage_fixed_bps(
    price: float,
    bps: float = 5.0,
) -> float:
    """Fixed basis-point slippage."""
    return price * bps / 10_000


def slippage_volume_dependent(
    price: float,
    order_qty: int,
    market_volume: int,
    eta: float = 0.1,
) -> float:
    """Volume-dependent slippage using square-root market-impact law.

    cost = eta * sigma_daily * price * sqrt(order_qty / market_volume)
    Simplified: uses eta * price * sqrt(participation_rate).
    """
    if market_volume <= 0:
        return price * eta
    participation = order_qty / market_volume
    return price * eta * math.sqrt(participation)


def slippage_volatility_adjusted(
    price: float,
    realised_vol: float,
    base_bps: float = 3.0,
    vol_multiplier: float = 50.0,
) -> float:
    """Slippage that scales with realised volatility.

    cost = price * (base_bps/10000 + vol_multiplier * realised_vol / 10000)
    """
    return price * (base_bps + vol_multiplier * realised_vol) / 10_000


def slippage_sqrt_impact(
    price: float,
    order_qty: int,
    market_volume: int,
    daily_vol: float = 0.01,
    impact_coeff: float = 0.5,
) -> float:
    """Almgren-Chriss square-root temporary impact model.

    cost = impact_coeff * daily_vol * price * sqrt(order_qty / market_volume)
    """
    if market_volume <= 0:
        return price * impact_coeff * daily_vol
    return impact_coeff * daily_vol * price * math.sqrt(order_qty / market_volume)


def estimate_bid_ask_cost(
    price: float,
    spread_bps: float = 10.0,
) -> float:
    """Half-spread crossing cost."""
    return price * spread_bps / 10_000 / 2.0


MODEL_FUNCTIONS = {
    "fixed_bps": slippage_fixed_bps,
    "volume_dependent": slippage_volume_dependent,
    "volatility_adjusted": slippage_volatility_adjusted,
    "sqrt_impact": slippage_sqrt_impact,
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ModelComparison:
    """Per-model fit metrics against observed slippage."""

    model_name: str
    mean_predicted: float
    mean_observed: float
    mae: float
    rmse: float
    r_squared: float
    bias: float  # predicted - observed


@dataclass
class InstrumentProfile:
    """Slippage profile for a single instrument."""

    instrument: str
    n_trades: int
    avg_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    avg_spread_bps: float
    total_slippage_dollars: float


@dataclass
class TimeOfDayPattern:
    """Slippage pattern for a time-of-day bucket."""

    bucket: str  # e.g. "09:30-10:00"
    n_trades: int
    avg_slippage_bps: float
    median_slippage_bps: float


@dataclass
class CalibrationResult:
    """Calibration output from fitting models to historical data."""

    best_model: str
    calibrated_params: Dict[str, float]
    model_comparisons: List[ModelComparison]
    n_trades: int


@dataclass
class BudgetAllocation:
    """Slippage budget allocation across instruments."""

    instrument: str
    weight: float
    allocated_bps: float
    current_avg_bps: float
    budget_utilization: float  # current / allocated


@dataclass
class SlippageResult:
    """Full result from slippage analysis."""

    instrument_profiles: List[InstrumentProfile]
    time_patterns: List[TimeOfDayPattern]
    calibration: CalibrationResult
    budget_allocations: List[BudgetAllocation]
    total_slippage_dollars: float
    avg_slippage_bps: float
    n_trades: int


# ── Calibration engine ───────────────────────────────────────────────────


def calibrate_models(
    trades: pd.DataFrame,
    observed_col: str = "observed_slippage",
) -> CalibrationResult:
    """Fit all models to observed slippage and rank by RMSE.

    Expected columns: price, order_qty, market_volume, realised_vol,
                      spread_bps, observed_slippage.
    """
    if trades.empty or observed_col not in trades.columns:
        return CalibrationResult(
            best_model="fixed_bps",
            calibrated_params={"bps": 5.0},
            model_comparisons=[],
            n_trades=0,
        )

    observed = trades[observed_col].values
    n = len(observed)

    comparisons: List[ModelComparison] = []

    # 1) Fixed bps — calibrate bps to match mean observed
    mean_obs_bps = float(np.mean(observed / trades["price"].values * 10_000))
    fixed_pred = np.array([
        slippage_fixed_bps(float(r["price"]), mean_obs_bps)
        for _, r in trades.iterrows()
    ])
    comparisons.append(_model_comparison("fixed_bps", fixed_pred, observed))

    # 2) Volume dependent
    vol_dep_pred = np.array([
        slippage_volume_dependent(
            float(r["price"]),
            int(r.get("order_qty", 1)),
            int(r.get("market_volume", 1000)),
        )
        for _, r in trades.iterrows()
    ])
    comparisons.append(_model_comparison("volume_dependent", vol_dep_pred, observed))

    # 3) Volatility adjusted
    vol_adj_pred = np.array([
        slippage_volatility_adjusted(
            float(r["price"]),
            float(r.get("realised_vol", 0.01)),
        )
        for _, r in trades.iterrows()
    ])
    comparisons.append(_model_comparison("volatility_adjusted", vol_adj_pred, observed))

    # 4) Sqrt impact
    sqrt_pred = np.array([
        slippage_sqrt_impact(
            float(r["price"]),
            int(r.get("order_qty", 1)),
            int(r.get("market_volume", 1000)),
            float(r.get("realised_vol", 0.01)),
        )
        for _, r in trades.iterrows()
    ])
    comparisons.append(_model_comparison("sqrt_impact", sqrt_pred, observed))

    # Sort by RMSE
    comparisons.sort(key=lambda c: c.rmse)
    best = comparisons[0].model_name

    params: Dict[str, float] = {"bps": mean_obs_bps}

    return CalibrationResult(
        best_model=best,
        calibrated_params=params,
        model_comparisons=comparisons,
        n_trades=n,
    )


def _model_comparison(
    name: str, predicted: np.ndarray, observed: np.ndarray
) -> ModelComparison:
    """Compute fit metrics."""
    residuals = predicted - observed
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    bias = float(np.mean(residuals))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return ModelComparison(
        model_name=name,
        mean_predicted=float(predicted.mean()),
        mean_observed=float(observed.mean()),
        mae=mae,
        rmse=rmse,
        r_squared=r2,
        bias=bias,
    )


# ── Instrument profiles ──────────────────────────────────────────────────


def build_instrument_profiles(
    trades: pd.DataFrame,
    instrument_col: str = "instrument",
    slippage_col: str = "observed_slippage",
    price_col: str = "price",
) -> List[InstrumentProfile]:
    """Build per-instrument slippage profiles."""
    if trades.empty or instrument_col not in trades.columns:
        return []

    profiles: List[InstrumentProfile] = []
    for inst, grp in trades.groupby(instrument_col):
        slip_bps = grp[slippage_col] / grp[price_col] * 10_000
        spread_bps = grp["spread_bps"] if "spread_bps" in grp.columns else pd.Series([10.0] * len(grp))

        profiles.append(InstrumentProfile(
            instrument=str(inst),
            n_trades=len(grp),
            avg_slippage_bps=float(slip_bps.mean()),
            median_slippage_bps=float(slip_bps.median()),
            p95_slippage_bps=float(np.percentile(slip_bps, 95)),
            avg_spread_bps=float(spread_bps.mean()),
            total_slippage_dollars=float(grp[slippage_col].sum()),
        ))

    return sorted(profiles, key=lambda p: p.total_slippage_dollars, reverse=True)


# ── Time-of-day patterns ────────────────────────────────────────────────


def build_time_patterns(
    trades: pd.DataFrame,
    time_col: str = "trade_time",
    slippage_col: str = "observed_slippage",
    price_col: str = "price",
) -> List[TimeOfDayPattern]:
    """Build slippage patterns by time-of-day bucket."""
    if trades.empty or time_col not in trades.columns:
        return []

    df = trades.copy()
    df["_time"] = pd.to_datetime(df[time_col])
    df["_hour"] = df["_time"].dt.hour
    df["_slip_bps"] = df[slippage_col] / df[price_col] * 10_000

    patterns: List[TimeOfDayPattern] = []
    for hour, grp in df.groupby("_hour"):
        bucket = f"{int(hour):02d}:00-{int(hour):02d}:59"
        patterns.append(TimeOfDayPattern(
            bucket=bucket,
            n_trades=len(grp),
            avg_slippage_bps=float(grp["_slip_bps"].mean()),
            median_slippage_bps=float(grp["_slip_bps"].median()),
        ))

    return sorted(patterns, key=lambda p: int(p.bucket[:2]))


# ── Budget allocation ────────────────────────────────────────────────────


def allocate_slippage_budget(
    profiles: List[InstrumentProfile],
    total_budget_bps: float = 20.0,
) -> List[BudgetAllocation]:
    """Allocate slippage budget proportionally to trade count."""
    if not profiles:
        return []

    total_trades = sum(p.n_trades for p in profiles)
    if total_trades == 0:
        return []

    allocations: List[BudgetAllocation] = []
    for p in profiles:
        weight = p.n_trades / total_trades
        allocated = total_budget_bps * weight
        util = p.avg_slippage_bps / allocated if allocated > 1e-12 else 0.0
        allocations.append(BudgetAllocation(
            instrument=p.instrument,
            weight=weight,
            allocated_bps=allocated,
            current_avg_bps=p.avg_slippage_bps,
            budget_utilization=util,
        ))

    return sorted(allocations, key=lambda a: a.budget_utilization, reverse=True)


# ── Core engine ──────────────────────────────────────────────────────────


class SlippageEngine:
    """Advanced slippage modeling and analysis engine.

    Args:
        total_budget_bps: Portfolio-wide slippage budget in bps.
    """

    def __init__(self, total_budget_bps: float = 20.0):
        if total_budget_bps <= 0:
            raise ValueError("total_budget_bps must be positive")
        self.total_budget_bps = total_budget_bps

    def analyze(self, trades: pd.DataFrame) -> SlippageResult:
        """Run full slippage analysis.

        Expected columns:
            price, order_qty, market_volume, realised_vol, observed_slippage.
        Optional: instrument, trade_time, spread_bps.
        """
        required = {"price", "observed_slippage"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Fill defaults for optional columns
        df = trades.copy()
        if "order_qty" not in df.columns:
            df["order_qty"] = 1
        if "market_volume" not in df.columns:
            df["market_volume"] = 1000
        if "realised_vol" not in df.columns:
            df["realised_vol"] = 0.01
        if "spread_bps" not in df.columns:
            df["spread_bps"] = 10.0

        calibration = calibrate_models(df)
        profiles = build_instrument_profiles(df)
        time_patterns = build_time_patterns(df)
        budget = allocate_slippage_budget(profiles, self.total_budget_bps)

        total_slip = float(df["observed_slippage"].sum())
        avg_bps = float(
            (df["observed_slippage"] / df["price"] * 10_000).mean()
        ) if len(df) > 0 else 0.0

        return SlippageResult(
            instrument_profiles=profiles,
            time_patterns=time_patterns,
            calibration=calibration,
            budget_allocations=budget,
            total_slippage_dollars=total_slip,
            avg_slippage_bps=avg_bps,
            n_trades=len(df),
        )

    @staticmethod
    def generate_report(
        result: SlippageResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fmt_bps(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _model_comparison_table(comps: List[ModelComparison]) -> str:
    if not comps:
        return "<p class='meta'>No calibration data.</p>"
    rows = ""
    for i, c in enumerate(comps):
        cls = "best-row" if i == 0 else ""
        rows += f"""<tr class="{cls}">
          <td style="text-align:left">{c.model_name}</td>
          <td>{_fmt_dollar(c.mean_predicted)}</td>
          <td>{_fmt_dollar(c.mean_observed)}</td>
          <td>{_fmt_dollar(c.mae)}</td>
          <td>{_fmt_dollar(c.rmse)}</td>
          <td>{c.r_squared:.3f}</td>
          <td>{_fmt_dollar(c.bias)}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr><th style="text-align:left">Model</th><th>Mean Pred</th><th>Mean Obs</th>
          <th>MAE</th><th>RMSE</th><th>R²</th><th>Bias</th></tr>
      {rows}
    </table>"""


def _instrument_table(profiles: List[InstrumentProfile]) -> str:
    if not profiles:
        return "<p class='meta'>No instrument data.</p>"
    rows = ""
    for p in profiles:
        rows += f"""<tr>
          <td style="text-align:left">{p.instrument}</td>
          <td>{p.n_trades}</td>
          <td>{_fmt_bps(p.avg_slippage_bps)}</td>
          <td>{_fmt_bps(p.median_slippage_bps)}</td>
          <td>{_fmt_bps(p.p95_slippage_bps)}</td>
          <td>{_fmt_bps(p.avg_spread_bps)}</td>
          <td>{_fmt_dollar(p.total_slippage_dollars)}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr><th style="text-align:left">Instrument</th><th>Trades</th>
          <th>Avg (bps)</th><th>Med (bps)</th><th>P95 (bps)</th>
          <th>Spread (bps)</th><th>Total $</th></tr>
      {rows}
    </table>"""


def _time_pattern_svg(patterns: List[TimeOfDayPattern]) -> str:
    if not patterns:
        return "<p class='meta'>No time-of-day data.</p>"

    w, h = 700, 250
    pad = 60
    n = len(patterns)
    values = [p.avg_slippage_bps for p in patterns]
    max_v = max(values) * 1.2 if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    bar_w = (w - 2 * pad) / n * 0.7
    gap = (w - 2 * pad) / n

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">'
        f"Slippage by Time of Day (bps)</text>"
    )

    for i, p in enumerate(patterns):
        x = pad + i * gap + (gap - bar_w) / 2
        bh = (p.avg_slippage_bps / max_v) * (h - 80)
        y = h - 40 - bh
        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" '
            f'height="{bh:.0f}" fill="#58a6ff" rx="3" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.0f}" y="{y - 4:.0f}" text-anchor="middle" '
            f'font-size="9" fill="#c9d1d9">{p.avg_slippage_bps:.1f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.0f}" y="{h - 20:.0f}" text-anchor="middle" '
            f'font-size="8" fill="#8b949e">{p.bucket[:5]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _budget_table(allocs: List[BudgetAllocation]) -> str:
    if not allocs:
        return "<p class='meta'>No budget data.</p>"
    rows = ""
    for a in allocs:
        util_cls = "over" if a.budget_utilization > 1.0 else "ok"
        rows += f"""<tr>
          <td style="text-align:left">{a.instrument}</td>
          <td>{a.weight:.1%}</td>
          <td>{_fmt_bps(a.allocated_bps)}</td>
          <td>{_fmt_bps(a.current_avg_bps)}</td>
          <td class="{util_cls}">{a.budget_utilization:.1%}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr><th style="text-align:left">Instrument</th><th>Weight</th>
          <th>Budget (bps)</th><th>Current (bps)</th><th>Utilization</th></tr>
      {rows}
    </table>"""


def _model_comparison_svg(comps: List[ModelComparison]) -> str:
    """Bar chart comparing model RMSE."""
    if not comps:
        return ""
    w, h = 500, 200
    pad = 100
    n = len(comps)
    max_rmse = max(c.rmse for c in comps) * 1.2
    if max_rmse <= 0:
        max_rmse = 1.0

    bar_h = 28
    gap = 8
    chart_h = n * (bar_h + gap) + 40

    parts = [f'<svg viewBox="0 0 {w} {chart_h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="18" text-anchor="middle" class="svg-title">'
        f"Model RMSE Comparison</text>"
    )

    bar_area = w - pad - 40
    for i, c in enumerate(comps):
        y = 30 + i * (bar_h + gap)
        bw = (c.rmse / max_rmse) * bar_area
        color = "#3fb950" if i == 0 else "#58a6ff"
        parts.append(
            f'<text x="{pad - 5}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" '
            f'font-size="10" fill="#8b949e">{c.model_name}</text>'
        )
        parts.append(
            f'<rect x="{pad}" y="{y}" width="{bw:.0f}" height="{bar_h}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{pad + bw + 4:.0f}" y="{y + bar_h * 0.7:.0f}" '
            f'font-size="9" fill="#c9d1d9">{_fmt_dollar(c.rmse)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(result: SlippageResult) -> str:
    cal = result.calibration
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Slippage Model Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.85em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.2em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .best-row td {{ color: #3fb950; font-weight: 600; }}
  .over {{ color: #f85149; font-weight: 700; }}
  .ok {{ color: #3fb950; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .svg-label {{ fill: #8b949e; font-size: 10px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Slippage Model Analysis</h1>
<p class="meta">{result.n_trades} trades analyzed &middot;
   Best model: {cal.best_model}</p>

<div class="summary">
  <div class="stat"><div class="label">Total Slippage</div>
    <div class="value">{_fmt_dollar(result.total_slippage_dollars)}</div></div>
  <div class="stat"><div class="label">Avg Slippage</div>
    <div class="value">{_fmt_bps(result.avg_slippage_bps)} bps</div></div>
  <div class="stat"><div class="label">Best Model</div>
    <div class="value">{cal.best_model}</div></div>
  <div class="stat"><div class="label">Best RMSE</div>
    <div class="value">{_fmt_dollar(cal.model_comparisons[0].rmse) if cal.model_comparisons else 'N/A'}</div></div>
</div>

<h2>Model Comparison</h2>
<div class="two-col">
  {_model_comparison_table(cal.model_comparisons)}
  {_model_comparison_svg(cal.model_comparisons)}
</div>

<h2>Instrument Profiles</h2>
{_instrument_table(result.instrument_profiles)}

<h2>Time-of-Day Patterns</h2>
{_time_pattern_svg(result.time_patterns)}

<h2>Slippage Budget Allocation</h2>
{_budget_table(result.budget_allocations)}

</body>
</html>"""
