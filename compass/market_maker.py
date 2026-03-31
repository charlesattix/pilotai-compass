"""
Market-making strategy simulator.

Components:
  - Avellaneda-Stoikov optimal quoting model
  - Bid-ask spread optimisation (volatility-regime adaptive)
  - Inventory management with position limits
  - Adverse selection detection (toxicity scoring)
  - PnL decomposition: spread capture, inventory risk, adverse selection
  - Fill-rate modelling
  - HTML report at reports/market_maker.html

This is READ-ONLY simulation.  No broker connections, no trade placement.

Usage::

    from compass.market_maker import MarketMakerSimulator
    sim = MarketMakerSimulator(config)
    result = sim.simulate(mid_prices)
    MarketMakerSimulator.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "market_maker.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class MMConfig:
    """Market-maker configuration."""

    gamma: float = 0.1           # risk aversion
    k: float = 1.5               # order-arrival intensity
    sigma: float = 0.01          # daily volatility
    dt: float = 1.0 / 252        # time step (1 trading day)
    T: float = 1.0               # horizon (1 year)
    position_limit: int = 100    # max absolute inventory
    min_spread_bps: float = 2.0  # minimum half-spread in bps
    lot_size: int = 1            # contracts per fill
    base_fill_prob: float = 0.3  # base probability of fill per step


@dataclass
class QuoteState:
    """State of quotes at a single time step."""

    step: int
    mid_price: float
    bid: float
    ask: float
    half_spread_bps: float
    inventory: int
    reservation_price: float
    optimal_spread: float


@dataclass
class FillEvent:
    """A single fill event."""

    step: int
    side: str       # "buy" or "sell"
    price: float
    quantity: int
    mid_price: float
    inventory_after: int


@dataclass
class PnLDecomposition:
    """Decompose total PnL into components."""

    total_pnl: float
    spread_capture: float
    inventory_risk: float
    adverse_selection: float


@dataclass
class AdverseSelectionMetrics:
    """Metrics for detecting toxic flow."""

    toxicity_score: float       # 0-1, higher = more toxic
    avg_adverse_move_bps: float
    pct_adverse_fills: float
    n_adverse_fills: int
    n_total_fills: int


@dataclass
class SpreadAnalysis:
    """Spread statistics."""

    avg_half_spread_bps: float
    median_half_spread_bps: float
    min_half_spread_bps: float
    max_half_spread_bps: float
    avg_effective_spread_bps: float


@dataclass
class SimulationResult:
    """Full result from market-maker simulation."""

    config: MMConfig
    quotes: List[QuoteState]
    fills: List[FillEvent]
    pnl_decomp: PnLDecomposition
    adverse_selection: AdverseSelectionMetrics
    spread_analysis: SpreadAnalysis
    inventory_path: np.ndarray
    pnl_path: np.ndarray
    fill_rate: float
    n_steps: int
    final_pnl: float
    max_inventory: int
    min_inventory: int


# ── Avellaneda-Stoikov model ─────────────────────────────────────────────


def reservation_price(
    mid: float,
    inventory: int,
    gamma: float,
    sigma: float,
    T_remaining: float,
) -> float:
    """Avellaneda-Stoikov reservation price.

    r = mid - inventory * gamma * sigma^2 * T_remaining
    """
    return mid - inventory * gamma * sigma ** 2 * T_remaining


def optimal_spread(
    gamma: float,
    sigma: float,
    T_remaining: float,
    k: float,
) -> float:
    """Avellaneda-Stoikov optimal spread.

    delta = gamma * sigma^2 * T_remaining + (2/gamma) * ln(1 + gamma/k)
    """
    vol_component = gamma * sigma ** 2 * T_remaining
    intensity_component = (2.0 / gamma) * math.log(1.0 + gamma / k)
    return vol_component + intensity_component


def compute_half_spread(
    mid: float,
    spread: float,
    min_spread_bps: float,
) -> float:
    """Half spread floored by minimum bps."""
    half = spread / 2.0
    min_half = mid * min_spread_bps / 10_000
    return max(half, min_half)


# ── Fill probability ─────────────────────────────────────────────────────


def fill_probability(
    half_spread: float,
    mid: float,
    k: float,
    base_prob: float,
) -> float:
    """Probability of fill decreases with wider spread.

    p = base_prob * exp(-k * half_spread / mid)
    """
    if mid <= 0:
        return 0.0
    return base_prob * math.exp(-k * half_spread / mid)


# ── Adverse selection ────────────────────────────────────────────────────


def detect_adverse_selection(
    fills: List[FillEvent],
    mid_prices: np.ndarray,
    lookforward: int = 5,
) -> AdverseSelectionMetrics:
    """Detect adverse selection by measuring price move after fills.

    A fill is "adverse" if the price moves against the market maker
    within `lookforward` steps.
    """
    if not fills:
        return AdverseSelectionMetrics(
            toxicity_score=0.0, avg_adverse_move_bps=0.0,
            pct_adverse_fills=0.0, n_adverse_fills=0, n_total_fills=0,
        )

    n_prices = len(mid_prices)
    adverse_moves: List[float] = []
    n_adverse = 0

    for f in fills:
        future_idx = min(f.step + lookforward, n_prices - 1)
        if future_idx <= f.step:
            continue

        future_mid = mid_prices[future_idx]
        move_bps = (future_mid - f.mid_price) / f.mid_price * 10_000

        # Adverse: bought and price dropped, or sold and price rose
        if f.side == "buy" and move_bps < 0:
            adverse_moves.append(abs(move_bps))
            n_adverse += 1
        elif f.side == "sell" and move_bps > 0:
            adverse_moves.append(abs(move_bps))
            n_adverse += 1

    n_total = len(fills)
    pct_adverse = n_adverse / n_total if n_total > 0 else 0.0
    avg_adverse = float(np.mean(adverse_moves)) if adverse_moves else 0.0

    # Toxicity: combination of fraction and magnitude
    toxicity = pct_adverse * min(1.0, avg_adverse / 20.0)

    return AdverseSelectionMetrics(
        toxicity_score=min(1.0, toxicity),
        avg_adverse_move_bps=avg_adverse,
        pct_adverse_fills=pct_adverse,
        n_adverse_fills=n_adverse,
        n_total_fills=n_total,
    )


# ── PnL decomposition ───────────────────────────────────────────────────


def decompose_pnl(
    fills: List[FillEvent],
    mid_prices: np.ndarray,
    final_inventory: int,
) -> PnLDecomposition:
    """Decompose PnL into spread capture, inventory risk, and adverse selection.

    - Spread capture: sum of (fill_price - mid) for sells, (mid - fill_price) for buys
    - Inventory risk: mark-to-market of final inventory position
    - Adverse selection: difference from total realized PnL
    """
    if not fills:
        return PnLDecomposition(
            total_pnl=0.0, spread_capture=0.0,
            inventory_risk=0.0, adverse_selection=0.0,
        )

    # Spread capture: half-spread earned per fill
    spread_capture = 0.0
    for f in fills:
        if f.side == "sell":
            spread_capture += (f.price - f.mid_price) * f.quantity * 100
        else:
            spread_capture += (f.mid_price - f.price) * f.quantity * 100

    # Inventory risk: mark-to-market final position
    final_mid = mid_prices[-1]
    # Average cost of remaining inventory
    cash_flow = 0.0
    net_qty = 0
    for f in fills:
        if f.side == "buy":
            cash_flow -= f.price * f.quantity * 100
            net_qty += f.quantity
        else:
            cash_flow += f.price * f.quantity * 100
            net_qty -= f.quantity

    inventory_mtm = net_qty * final_mid * 100  # mark to market
    total_pnl = cash_flow + inventory_mtm

    inventory_risk = total_pnl - spread_capture
    adverse_sel = min(0.0, inventory_risk * 0.5)  # portion attributed to toxicity
    inventory_risk -= adverse_sel

    return PnLDecomposition(
        total_pnl=total_pnl,
        spread_capture=spread_capture,
        inventory_risk=inventory_risk,
        adverse_selection=adverse_sel,
    )


# ── Core simulator ───────────────────────────────────────────────────────


class MarketMakerSimulator:
    """Simulates a market-making strategy with Avellaneda-Stoikov quoting."""

    def __init__(self, config: Optional[MMConfig] = None, seed: Optional[int] = None):
        self.config = config or MMConfig()
        self.seed = seed

    def simulate(self, mid_prices: pd.Series) -> SimulationResult:
        """Run simulation over a series of mid prices.

        Args:
            mid_prices: Series of mid prices (one per time step).
        """
        cfg = self.config
        rng = np.random.RandomState(self.seed)
        mids = mid_prices.values.astype(float)
        n = len(mids)

        if n < 2:
            raise ValueError("Need at least 2 mid prices")

        # Estimate realized vol from prices if needed
        log_rets = np.diff(np.log(mids))
        realized_sigma = float(np.std(log_rets)) * math.sqrt(252)
        sigma = realized_sigma if realized_sigma > 1e-6 else cfg.sigma

        inventory = 0
        quotes: List[QuoteState] = []
        fills: List[FillEvent] = []
        inventory_path = np.zeros(n, dtype=int)
        pnl_running = 0.0
        pnl_path = np.zeros(n)
        cash = 0.0

        for t in range(n):
            T_rem = max(cfg.T - t * cfg.dt, cfg.dt)

            # Avellaneda-Stoikov
            r_price = reservation_price(mids[t], inventory, cfg.gamma, sigma, T_rem)
            opt_spread = optimal_spread(cfg.gamma, sigma, T_rem, cfg.k)
            half = compute_half_spread(mids[t], opt_spread, cfg.min_spread_bps)

            bid = r_price - half
            ask = r_price + half
            hs_bps = half / mids[t] * 10_000

            quotes.append(QuoteState(
                step=t, mid_price=mids[t], bid=bid, ask=ask,
                half_spread_bps=hs_bps, inventory=inventory,
                reservation_price=r_price, optimal_spread=opt_spread,
            ))

            # Fill simulation
            bid_fill_prob = fill_probability(half, mids[t], cfg.k, cfg.base_fill_prob)
            ask_fill_prob = fill_probability(half, mids[t], cfg.k, cfg.base_fill_prob)

            # Buy fill (someone hits our bid)
            if inventory < cfg.position_limit and rng.random() < bid_fill_prob:
                fills.append(FillEvent(
                    step=t, side="buy", price=bid, quantity=cfg.lot_size,
                    mid_price=mids[t], inventory_after=inventory + cfg.lot_size,
                ))
                cash -= bid * cfg.lot_size * 100
                inventory += cfg.lot_size

            # Sell fill (someone lifts our ask)
            if inventory > -cfg.position_limit and rng.random() < ask_fill_prob:
                fills.append(FillEvent(
                    step=t, side="sell", price=ask, quantity=cfg.lot_size,
                    mid_price=mids[t], inventory_after=inventory - cfg.lot_size,
                ))
                cash += ask * cfg.lot_size * 100
                inventory -= cfg.lot_size

            inventory_path[t] = inventory
            pnl_path[t] = cash + inventory * mids[t] * 100

        # Analysis
        pnl_decomp = decompose_pnl(fills, mids, inventory)
        adverse = detect_adverse_selection(fills, mids)

        # Spread analysis
        spreads = [q.half_spread_bps for q in quotes]
        eff_spreads = []
        for f in fills:
            eff = abs(f.price - f.mid_price) / f.mid_price * 10_000
            eff_spreads.append(eff)

        spread_analysis = SpreadAnalysis(
            avg_half_spread_bps=float(np.mean(spreads)),
            median_half_spread_bps=float(np.median(spreads)),
            min_half_spread_bps=float(np.min(spreads)),
            max_half_spread_bps=float(np.max(spreads)),
            avg_effective_spread_bps=float(np.mean(eff_spreads)) if eff_spreads else 0.0,
        )

        fill_rate = len(fills) / (2 * n) if n > 0 else 0.0

        return SimulationResult(
            config=cfg,
            quotes=quotes,
            fills=fills,
            pnl_decomp=pnl_decomp,
            adverse_selection=adverse,
            spread_analysis=spread_analysis,
            inventory_path=inventory_path,
            pnl_path=pnl_path,
            fill_rate=fill_rate,
            n_steps=n,
            final_pnl=float(pnl_path[-1]),
            max_inventory=int(inventory_path.max()),
            min_inventory=int(inventory_path.min()),
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: SimulationResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_bps(v: float) -> str:
    return f"{v:.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _svg_line(
    values: np.ndarray, title: str, color: str = "#58a6ff",
    w: int = 700, h: int = 200,
) -> str:
    if len(values) < 2:
        return ""
    n = len(values)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65

    y_min = float(values.min())
    y_max = float(values.max())
    if y_max <= y_min:
        y_max = y_min + 1.0

    def tx(i: int) -> float:
        return pad + i / max(n - 1, 1) * pw

    def ty(v: float) -> float:
        return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>'
    )
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(
            f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" '
            f'stroke="#30363d" stroke-dasharray="3,3"/>'
        )
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(float(values[i])):.1f}"
        for i in range(n)
    )
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _pnl_waterfall(decomp: PnLDecomposition) -> str:
    items = [
        ("Spread Capture", decomp.spread_capture),
        ("Inventory Risk", decomp.inventory_risk),
        ("Adverse Selection", decomp.adverse_selection),
    ]
    w, h = 500, 220
    pad_l, pad_t = 130, 35
    bar_h = 30
    gap = 12
    vals = [v for _, v in items]
    abs_max = max(abs(v) for v in vals) if vals else 1.0
    if abs_max == 0:
        abs_max = 1.0
    bar_area = (w - pad_l - 50) / 2
    mid_x = pad_l + bar_area

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">'
        f"PnL Attribution</text>"
    )
    parts.append(
        f'<line x1="{mid_x:.0f}" y1="{pad_t}" x2="{mid_x:.0f}" y2="{h - 10}" '
        f'stroke="#30363d"/>'
    )

    for i, (label, val) in enumerate(items):
        y = pad_t + 5 + i * (bar_h + gap)
        bw = abs(val) / abs_max * bar_area
        color = "#3fb950" if val >= 0 else "#f85149"
        bx = mid_x if val >= 0 else mid_x - bw
        parts.append(
            f'<text x="{pad_l - 5}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" '
            f'font-size="10" fill="#8b949e">{label}</text>'
        )
        parts.append(
            f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{bx + bw + 4:.0f}" y="{y + bar_h * 0.7:.0f}" '
            f'font-size="9" fill="#c9d1d9">{_fmt_dollar(val)}</text>'
        )

    # Total bar
    y = pad_t + 5 + len(items) * (bar_h + gap)
    total = decomp.total_pnl
    bw = abs(total) / abs_max * bar_area
    color = "#58a6ff"
    bx = mid_x if total >= 0 else mid_x - bw
    parts.append(
        f'<text x="{pad_l - 5}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" '
        f'font-size="10" fill="#f0f6fc" font-weight="bold">Total</text>'
    )
    parts.append(
        f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" '
        f'fill="{color}" rx="3"/>'
    )
    parts.append(
        f'<text x="{bx + bw + 4:.0f}" y="{y + bar_h * 0.7:.0f}" '
        f'font-size="10" fill="#f0f6fc" font-weight="bold">{_fmt_dollar(total)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(result: SimulationResult) -> str:
    cfg = result.config
    d = result.pnl_decomp
    a = result.adverse_selection
    s = result.spread_analysis

    tox_color = "#3fb950" if a.toxicity_score < 0.3 else "#d29922" if a.toxicity_score < 0.6 else "#f85149"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Market Maker Simulation Report</title>
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
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.15em; }}
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
<h1>Market Maker Simulation</h1>
<p class="meta">{result.n_steps} steps &middot; {len(result.fills)} fills &middot;
   Fill rate: {_fmt_pct(result.fill_rate)} &middot;
   &gamma;={cfg.gamma}, k={cfg.k}, limit={cfg.position_limit}</p>

<div class="summary">
  <div class="stat"><div class="label">Final PnL</div>
    <div class="value">{_fmt_dollar(result.final_pnl)}</div></div>
  <div class="stat"><div class="label">Spread Capture</div>
    <div class="value">{_fmt_dollar(d.spread_capture)}</div></div>
  <div class="stat"><div class="label">Inventory Risk</div>
    <div class="value">{_fmt_dollar(d.inventory_risk)}</div></div>
  <div class="stat"><div class="label">Adverse Sel.</div>
    <div class="value">{_fmt_dollar(d.adverse_selection)}</div></div>
  <div class="stat"><div class="label">Avg Spread</div>
    <div class="value">{_fmt_bps(s.avg_half_spread_bps)} bps</div></div>
  <div class="stat"><div class="label">Toxicity</div>
    <div class="value" style="color:{tox_color}">{a.toxicity_score:.2f}</div></div>
  <div class="stat"><div class="label">Max Inventory</div>
    <div class="value">{result.max_inventory}</div></div>
  <div class="stat"><div class="label">Min Inventory</div>
    <div class="value">{result.min_inventory}</div></div>
</div>

<h2>PnL Attribution</h2>
{_pnl_waterfall(d)}

<h2>PnL Path</h2>
{_svg_line(result.pnl_path, "Cumulative PnL ($)", "#3fb950")}

<h2>Inventory Path</h2>
{_svg_line(result.inventory_path.astype(float), "Inventory Position", "#d29922")}

<div class="two-col">
  <div class="card">
    <h3>Spread Analysis</h3>
    <div class="metrics-grid">
      <div><span class="label">Avg Half-Spread</span><span class="value">{_fmt_bps(s.avg_half_spread_bps)} bps</span></div>
      <div><span class="label">Median</span><span class="value">{_fmt_bps(s.median_half_spread_bps)} bps</span></div>
      <div><span class="label">Min</span><span class="value">{_fmt_bps(s.min_half_spread_bps)} bps</span></div>
      <div><span class="label">Max</span><span class="value">{_fmt_bps(s.max_half_spread_bps)} bps</span></div>
      <div><span class="label">Effective</span><span class="value">{_fmt_bps(s.avg_effective_spread_bps)} bps</span></div>
      <div><span class="label">Fill Rate</span><span class="value">{_fmt_pct(result.fill_rate)}</span></div>
    </div>
  </div>
  <div class="card">
    <h3>Adverse Selection</h3>
    <div class="metrics-grid">
      <div><span class="label">Toxicity Score</span><span class="value" style="color:{tox_color}">{a.toxicity_score:.3f}</span></div>
      <div><span class="label">Avg Adverse Move</span><span class="value">{_fmt_bps(a.avg_adverse_move_bps)} bps</span></div>
      <div><span class="label">% Adverse Fills</span><span class="value">{_fmt_pct(a.pct_adverse_fills)}</span></div>
      <div><span class="label">Adverse Fills</span><span class="value">{a.n_adverse_fills}</span></div>
      <div><span class="label">Total Fills</span><span class="value">{a.n_total_fills}</span></div>
    </div>
  </div>
</div>

</body>
</html>"""
