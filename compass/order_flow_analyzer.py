"""
Order flow analysis engine.

Components:
  - Trade flow imbalance (buy vs sell volume ratio)
  - Cumulative delta tracking (running buy - sell volume)
  - Volume profile: POC (point of control), value area, HVN/LVN
  - VWAP with standard-deviation bands
  - Footprint chart data generation (price-level buy/sell volume)
  - Order flow divergence signals (price up + delta down = bearish)
  - Large trade detection with configurable threshold

HTML report at reports/order_flow.html with volume profile chart,
delta chart, flow metrics dashboard.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.order_flow_analyzer import OrderFlowAnalyzer
    analyzer = OrderFlowAnalyzer(trades_df)
    result = analyzer.analyze()
    OrderFlowAnalyzer.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "order_flow.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class FlowImbalance:
    """Trade flow imbalance metrics."""

    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance_ratio: float   # (buy - sell) / total, range [-1, 1]
    buy_trade_count: int
    sell_trade_count: int


@dataclass
class CumulativeDelta:
    """Cumulative delta (running buy - sell volume)."""

    values: np.ndarray
    timestamps: np.ndarray
    final_delta: float
    max_delta: float
    min_delta: float


@dataclass
class PriceLevel:
    """Volume at a single price level for volume profile."""

    price: float
    buy_volume: float
    sell_volume: float
    total_volume: float
    delta: float


@dataclass
class VolumeProfile:
    """Full volume profile with POC and value area."""

    levels: List[PriceLevel]
    poc_price: float           # point of control (highest volume price)
    poc_volume: float
    value_area_high: float     # 70% of volume between VAH and VAL
    value_area_low: float
    high_volume_nodes: List[float]   # HVN prices
    low_volume_nodes: List[float]    # LVN prices


@dataclass
class VWAPData:
    """VWAP with standard deviation bands."""

    vwap: np.ndarray
    upper_1sd: np.ndarray
    lower_1sd: np.ndarray
    upper_2sd: np.ndarray
    lower_2sd: np.ndarray
    timestamps: np.ndarray


@dataclass
class FootprintBar:
    """Footprint chart data for one time bar."""

    timestamp: Any
    open_price: float
    close_price: float
    levels: List[PriceLevel]
    bar_delta: float
    bar_volume: float


@dataclass
class DivergenceSignal:
    """Order flow divergence signal."""

    timestamp: Any
    signal_type: str   # "bearish_divergence" or "bullish_divergence"
    price_change: float
    delta_change: float
    strength: float    # 0-1


@dataclass
class LargeTrade:
    """Detected large trade."""

    timestamp: Any
    price: float
    volume: float
    side: str
    volume_multiple: float   # how many times the threshold


@dataclass
class AnalysisResult:
    """Full result from order flow analysis."""

    flow_imbalance: FlowImbalance
    cumulative_delta: CumulativeDelta
    volume_profile: VolumeProfile
    vwap_data: VWAPData
    footprint_bars: List[FootprintBar]
    divergence_signals: List[DivergenceSignal]
    large_trades: List[LargeTrade]
    n_trades: int
    time_range: Tuple[Any, Any]


# ── Flow imbalance ───────────────────────────────────────────────────────


def compute_flow_imbalance(
    sides: np.ndarray,
    volumes: np.ndarray,
) -> FlowImbalance:
    """Compute buy vs sell flow imbalance.

    Args:
        sides: array of 1 (buy) or -1 (sell).
        volumes: array of trade volumes.
    """
    buy_mask = sides > 0
    sell_mask = sides < 0
    buy_vol = float(volumes[buy_mask].sum())
    sell_vol = float(volumes[sell_mask].sum())
    total = buy_vol + sell_vol

    imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0

    return FlowImbalance(
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        total_volume=total,
        imbalance_ratio=imbalance,
        buy_trade_count=int(buy_mask.sum()),
        sell_trade_count=int(sell_mask.sum()),
    )


# ── Cumulative delta ─────────────────────────────────────────────────────


def compute_cumulative_delta(
    sides: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
) -> CumulativeDelta:
    """Track cumulative delta (running buy - sell volume)."""
    signed = sides * volumes
    cum = np.cumsum(signed)
    return CumulativeDelta(
        values=cum,
        timestamps=timestamps,
        final_delta=float(cum[-1]) if len(cum) > 0 else 0.0,
        max_delta=float(cum.max()) if len(cum) > 0 else 0.0,
        min_delta=float(cum.min()) if len(cum) > 0 else 0.0,
    )


# ── Volume profile ──────────────────────────────────────────────────────


def compute_volume_profile(
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    n_levels: int = 50,
) -> VolumeProfile:
    """Build volume profile with POC, value area, HVN/LVN.

    Args:
        prices: trade prices.
        volumes: trade volumes.
        sides: 1 (buy) or -1 (sell).
        n_levels: number of price bins.
    """
    if len(prices) == 0:
        return VolumeProfile(
            levels=[], poc_price=0.0, poc_volume=0.0,
            value_area_high=0.0, value_area_low=0.0,
            high_volume_nodes=[], low_volume_nodes=[],
        )

    p_min, p_max = float(prices.min()), float(prices.max())
    if p_max <= p_min:
        p_max = p_min + 0.01

    edges = np.linspace(p_min, p_max, n_levels + 1)
    bin_idx = np.digitize(prices, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_levels - 1)

    levels: List[PriceLevel] = []
    for i in range(n_levels):
        mask = bin_idx == i
        if not mask.any():
            continue
        mid_price = (edges[i] + edges[i + 1]) / 2
        bv = float(volumes[mask & (sides > 0)].sum())
        sv = float(volumes[mask & (sides < 0)].sum())
        levels.append(PriceLevel(
            price=mid_price, buy_volume=bv, sell_volume=sv,
            total_volume=bv + sv, delta=bv - sv,
        ))

    if not levels:
        return VolumeProfile(
            levels=[], poc_price=0.0, poc_volume=0.0,
            value_area_high=0.0, value_area_low=0.0,
            high_volume_nodes=[], low_volume_nodes=[],
        )

    # POC
    poc = max(levels, key=lambda l: l.total_volume)

    # Value area (70% of volume)
    total_vol = sum(l.total_volume for l in levels)
    target = total_vol * 0.70
    sorted_levels = sorted(levels, key=lambda l: l.total_volume, reverse=True)
    accumulated = 0.0
    va_prices: List[float] = []
    for l in sorted_levels:
        accumulated += l.total_volume
        va_prices.append(l.price)
        if accumulated >= target:
            break
    va_high = max(va_prices) if va_prices else p_max
    va_low = min(va_prices) if va_prices else p_min

    # HVN / LVN (top/bottom quartile by volume)
    vol_values = [l.total_volume for l in levels]
    if len(vol_values) >= 4:
        q75 = np.percentile(vol_values, 75)
        q25 = np.percentile(vol_values, 25)
        hvn = [l.price for l in levels if l.total_volume >= q75]
        lvn = [l.price for l in levels if l.total_volume <= q25]
    else:
        hvn = [poc.price]
        lvn = []

    return VolumeProfile(
        levels=levels,
        poc_price=poc.price,
        poc_volume=poc.total_volume,
        value_area_high=va_high,
        value_area_low=va_low,
        high_volume_nodes=hvn,
        low_volume_nodes=lvn,
    )


# ── VWAP with bands ─────────────────────────────────────────────────────


def compute_vwap(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
) -> VWAPData:
    """VWAP with 1SD and 2SD bands."""
    n = len(prices)
    if n == 0:
        empty = np.array([])
        return VWAPData(
            vwap=empty, upper_1sd=empty, lower_1sd=empty,
            upper_2sd=empty, lower_2sd=empty, timestamps=empty,
        )

    cum_pv = np.cumsum(prices * volumes)
    cum_v = np.cumsum(volumes)
    cum_v_safe = np.where(cum_v > 0, cum_v, 1)
    vwap = cum_pv / cum_v_safe

    # Running variance of price around VWAP
    cum_pv2 = np.cumsum(prices**2 * volumes)
    variance = cum_pv2 / cum_v_safe - vwap**2
    variance = np.maximum(variance, 0)
    sd = np.sqrt(variance)

    return VWAPData(
        vwap=vwap,
        upper_1sd=vwap + sd,
        lower_1sd=vwap - sd,
        upper_2sd=vwap + 2 * sd,
        lower_2sd=vwap - 2 * sd,
        timestamps=timestamps,
    )


# ── Footprint bars ──────────────────────────────────────────────────────


def build_footprint_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    timestamps: np.ndarray,
    bar_size: int = 50,
    n_price_levels: int = 10,
) -> List[FootprintBar]:
    """Build footprint chart bars from trade data.

    Args:
        bar_size: number of trades per bar.
        n_price_levels: price levels within each bar.
    """
    n = len(prices)
    bars: List[FootprintBar] = []

    for start in range(0, n, bar_size):
        end = min(start + bar_size, n)
        bar_p = prices[start:end]
        bar_v = volumes[start:end]
        bar_s = sides[start:end]

        if len(bar_p) == 0:
            continue

        vp = compute_volume_profile(bar_p, bar_v, bar_s, n_price_levels)

        bars.append(FootprintBar(
            timestamp=timestamps[start],
            open_price=float(bar_p[0]),
            close_price=float(bar_p[-1]),
            levels=vp.levels,
            bar_delta=float((bar_s * bar_v).sum()),
            bar_volume=float(bar_v.sum()),
        ))

    return bars


# ── Divergence signals ───────────────────────────────────────────────────


def detect_divergences(
    prices: np.ndarray,
    cum_delta: np.ndarray,
    window: int = 20,
    threshold: float = 0.3,
) -> List[DivergenceSignal]:
    """Detect order flow divergence signals.

    Bearish: price rising but delta falling.
    Bullish: price falling but delta rising.
    """
    n = len(prices)
    signals: List[DivergenceSignal] = []

    for i in range(window, n):
        price_chg = (prices[i] - prices[i - window]) / prices[i - window]
        delta_chg = cum_delta[i] - cum_delta[i - window]
        # Normalise delta change relative to range
        delta_range = cum_delta[max(0, i - window):i + 1]
        d_range = delta_range.max() - delta_range.min()
        norm_delta = delta_chg / d_range if d_range > 1e-12 else 0.0

        if price_chg > 0.005 and norm_delta < -threshold:
            strength = min(1.0, abs(norm_delta) * abs(price_chg) * 100)
            signals.append(DivergenceSignal(
                timestamp=i, signal_type="bearish_divergence",
                price_change=price_chg, delta_change=delta_chg,
                strength=strength,
            ))
        elif price_chg < -0.005 and norm_delta > threshold:
            strength = min(1.0, abs(norm_delta) * abs(price_chg) * 100)
            signals.append(DivergenceSignal(
                timestamp=i, signal_type="bullish_divergence",
                price_change=price_chg, delta_change=delta_chg,
                strength=strength,
            ))

    return signals


# ── Large trade detection ────────────────────────────────────────────────


def detect_large_trades(
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    timestamps: np.ndarray,
    threshold_multiple: float = 3.0,
) -> List[LargeTrade]:
    """Detect trades exceeding threshold_multiple * median volume."""
    if len(volumes) == 0:
        return []

    median_vol = float(np.median(volumes))
    if median_vol <= 0:
        return []

    threshold = median_vol * threshold_multiple
    large: List[LargeTrade] = []

    for i in range(len(volumes)):
        if volumes[i] >= threshold:
            large.append(LargeTrade(
                timestamp=timestamps[i],
                price=float(prices[i]),
                volume=float(volumes[i]),
                side="buy" if sides[i] > 0 else "sell",
                volume_multiple=float(volumes[i] / median_vol),
            ))

    return large


# ── Core analyzer ────────────────────────────────────────────────────────


class OrderFlowAnalyzer:
    """Order flow analysis engine.

    Args:
        trades: DataFrame with columns: price, volume, side (1/-1), timestamp.
        footprint_bar_size: trades per footprint bar.
        divergence_window: lookback for divergence detection.
        large_trade_multiple: threshold multiple for large trades.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        footprint_bar_size: int = 50,
        divergence_window: int = 20,
        large_trade_multiple: float = 3.0,
    ):
        required = {"price", "volume", "side"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if trades.empty:
            raise ValueError("trades DataFrame must not be empty")

        self.trades = trades.copy()
        self.footprint_bar_size = footprint_bar_size
        self.divergence_window = divergence_window
        self.large_trade_multiple = large_trade_multiple

        # Extract arrays
        self.prices = trades["price"].values.astype(float)
        self.volumes = trades["volume"].values.astype(float)
        self.sides = trades["side"].values.astype(float)
        self.timestamps = (
            trades["timestamp"].values if "timestamp" in trades.columns
            else np.arange(len(trades))
        )

    def analyze(self) -> AnalysisResult:
        """Run full order flow analysis."""
        imbalance = compute_flow_imbalance(self.sides, self.volumes)
        cum_delta = compute_cumulative_delta(
            self.sides, self.volumes, self.timestamps
        )
        vol_profile = compute_volume_profile(
            self.prices, self.volumes, self.sides
        )
        vwap = compute_vwap(self.prices, self.volumes, self.timestamps)
        footprints = build_footprint_bars(
            self.prices, self.volumes, self.sides, self.timestamps,
            self.footprint_bar_size,
        )
        divergences = detect_divergences(
            self.prices, cum_delta.values, self.divergence_window,
        )
        large = detect_large_trades(
            self.prices, self.volumes, self.sides, self.timestamps,
            self.large_trade_multiple,
        )

        t_start = self.timestamps[0] if len(self.timestamps) > 0 else None
        t_end = self.timestamps[-1] if len(self.timestamps) > 0 else None

        return AnalysisResult(
            flow_imbalance=imbalance,
            cumulative_delta=cum_delta,
            volume_profile=vol_profile,
            vwap_data=vwap,
            footprint_bars=footprints,
            divergence_signals=divergences,
            large_trades=large,
            n_trades=len(self.trades),
            time_range=(t_start, t_end),
        )

    @staticmethod
    def generate_report(
        result: AnalysisResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}"


def _svg_line(values: np.ndarray, title: str, color: str = "#58a6ff",
              w: int = 700, h: int = 200) -> str:
    if len(values) < 2:
        return ""
    n = len(values)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65
    y_min, y_max = float(values.min()), float(values.max())
    if y_max <= y_min:
        y_max = y_min + 1.0

    def tx(i): return pad + i / max(n - 1, 1) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(float(values[i])):.1f}" for i in range(n))
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _volume_profile_svg(vp: VolumeProfile) -> str:
    if not vp.levels:
        return "<p class='meta'>No volume profile data.</p>"

    w, h = 400, 350
    pad_l, pad_b = 70, 30
    n = len(vp.levels)
    max_vol = max(l.total_volume for l in vp.levels)
    if max_vol <= 0:
        max_vol = 1.0

    bar_h = max(2, (h - 40 - pad_b) / n)
    bar_area = w - pad_l - 20

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="18" text-anchor="middle" class="svg-title">Volume Profile</text>')

    for i, l in enumerate(vp.levels):
        y = 28 + i * bar_h
        buy_w = (l.buy_volume / max_vol) * bar_area
        sell_w = (l.sell_volume / max_vol) * bar_area

        # Buy bars (right of center)
        parts.append(f'<rect x="{pad_l}" y="{y:.0f}" width="{buy_w:.0f}" height="{bar_h - 1:.0f}" fill="#3fb950" opacity="0.7" rx="1"/>')
        # Sell bars (also from left, stacked)
        parts.append(f'<rect x="{pad_l + buy_w:.0f}" y="{y:.0f}" width="{sell_w:.0f}" height="{bar_h - 1:.0f}" fill="#f85149" opacity="0.7" rx="1"/>')

        # POC highlight
        if abs(l.price - vp.poc_price) < 0.01:
            parts.append(f'<rect x="{pad_l}" y="{y:.0f}" width="{bar_area}" height="{bar_h:.0f}" fill="#58a6ff" opacity="0.2"/>')

        # Price label (every few levels)
        if i % max(1, n // 8) == 0:
            parts.append(f'<text x="{pad_l - 5}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" font-size="8" fill="#8b949e">{l.price:.2f}</text>')

    # POC line label
    parts.append(f'<text x="{w - 15}" y="{h - 10}" font-size="8" fill="#58a6ff" text-anchor="end">POC: {vp.poc_price:.2f}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _flow_metrics_card(fi: FlowImbalance) -> str:
    imb_color = "#3fb950" if fi.imbalance_ratio > 0.1 else "#f85149" if fi.imbalance_ratio < -0.1 else "#8b949e"
    return f"""
    <div class="card">
      <h3>Flow Metrics</h3>
      <div class="metrics-grid">
        <div><span class="label">Buy Volume</span><span class="value">{_fmt(fi.buy_volume, 0)}</span></div>
        <div><span class="label">Sell Volume</span><span class="value">{_fmt(fi.sell_volume, 0)}</span></div>
        <div><span class="label">Imbalance</span><span class="value" style="color:{imb_color}">{fi.imbalance_ratio:+.3f}</span></div>
        <div><span class="label">Buy Trades</span><span class="value">{fi.buy_trade_count}</span></div>
        <div><span class="label">Sell Trades</span><span class="value">{fi.sell_trade_count}</span></div>
        <div><span class="label">Total Volume</span><span class="value">{_fmt(fi.total_volume, 0)}</span></div>
      </div>
    </div>"""


def _build_html(result: AnalysisResult) -> str:
    fi = result.flow_imbalance
    cd = result.cumulative_delta
    vp = result.volume_profile

    n_bearish = sum(1 for d in result.divergence_signals if d.signal_type == "bearish_divergence")
    n_bullish = sum(1 for d in result.divergence_signals if d.signal_type == "bullish_divergence")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Order Flow Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Order Flow Analysis</h1>
<p class="meta">{result.n_trades} trades &middot;
   POC: {vp.poc_price:.2f} &middot;
   Final Delta: {_fmt(cd.final_delta, 0)} &middot;
   {len(result.large_trades)} large trades</p>

<div class="summary">
  <div class="stat"><div class="label">Imbalance</div>
    <div class="value">{fi.imbalance_ratio:+.3f}</div></div>
  <div class="stat"><div class="label">Final Delta</div>
    <div class="value">{_fmt(cd.final_delta, 0)}</div></div>
  <div class="stat"><div class="label">POC</div>
    <div class="value">{vp.poc_price:.2f}</div></div>
  <div class="stat"><div class="label">VA High</div>
    <div class="value">{vp.value_area_high:.2f}</div></div>
  <div class="stat"><div class="label">VA Low</div>
    <div class="value">{vp.value_area_low:.2f}</div></div>
  <div class="stat"><div class="label">Bearish Div</div>
    <div class="value">{n_bearish}</div></div>
  <div class="stat"><div class="label">Bullish Div</div>
    <div class="value">{n_bullish}</div></div>
  <div class="stat"><div class="label">Large Trades</div>
    <div class="value">{len(result.large_trades)}</div></div>
</div>

<div class="two-col">
  {_flow_metrics_card(fi)}
  {_volume_profile_svg(vp)}
</div>

<h2>Cumulative Delta</h2>
{_svg_line(cd.values, "Cumulative Delta", "#d29922")}

<h2>VWAP</h2>
{_svg_line(result.vwap_data.vwap, "VWAP", "#3fb950")}

</body>
</html>"""
