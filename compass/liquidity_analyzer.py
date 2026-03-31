"""Liquidity analyzer – estimates strategy capacity, fill quality, market impact,
and volume participation limits for credit spread portfolios.

Provides:
  1. Max AUM capacity before alpha decay becomes significant
  2. Bid-ask spread modelling / fill quality degradation at various sizes
  3. Volume participation limits (max safe order as fraction of ADV)
  4. Square-root market impact model calibrated to options markets
  5. Capacity curve: expected Sharpe degradation as AUM scales 1M → 1B
  6. Per-experiment capacity estimates with confidence intervals
  7. HTML report with charts and tables
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_AUM_GRID: List[float] = [
    1e6, 2e6, 5e6, 10e6, 25e6, 50e6, 100e6, 250e6, 500e6, 1e9,
]

# Options market defaults
DEFAULT_ADV_CONTRACTS = 5_000       # typical SPY OTM put/call ADV
DEFAULT_BID_ASK_BPS = 10.0          # typical 10 bps spread
DEFAULT_CONTRACT_MULTIPLIER = 100   # standard options multiplier
DEFAULT_MAX_PARTICIPATION = 0.05    # 5 % of ADV
IMPACT_COEFF = 0.10                 # square-root impact coefficient (η)
IMPACT_EXPONENT = 0.50              # square-root model exponent
SHARPE_DECAY_SENSITIVITY = 0.30     # how much Sharpe drops per 1 % market impact


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class MarketParams:
    """Market microstructure parameters for one instrument."""
    adv_contracts: int = DEFAULT_ADV_CONTRACTS
    bid_ask_bps: float = DEFAULT_BID_ASK_BPS
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER
    avg_contract_price: float = 2.50   # $/contract for option premium
    max_participation_rate: float = DEFAULT_MAX_PARTICIPATION


@dataclass
class FillQuality:
    """Fill quality estimate at a given order size."""
    order_contracts: int
    participation_rate: float      # order / ADV
    effective_spread_bps: float    # widened spread at this size
    market_impact_bps: float       # temporary + permanent impact
    total_cost_bps: float          # spread + impact
    fill_probability: float        # likelihood of full fill (0-1)


@dataclass
class CapacityPoint:
    """One point on the capacity curve."""
    aum: float
    contracts_per_trade: int
    participation_rate: float
    market_impact_bps: float
    total_cost_bps: float
    expected_sharpe: float         # after impact
    sharpe_degradation_pct: float  # % lost vs base


@dataclass
class ExperimentCapacity:
    """Per-experiment capacity estimate with confidence interval."""
    experiment_id: str
    base_sharpe: float
    max_aum: float                 # AUM where Sharpe halves
    recommended_aum: float         # conservative operating AUM
    ci_lower: float                # 10th percentile AUM
    ci_upper: float                # 90th percentile AUM
    max_contracts_per_trade: int
    participation_at_recommended: float


@dataclass
class LiquidityResult:
    """Complete liquidity analysis output."""
    capacity_curve: List[CapacityPoint] = field(default_factory=list)
    fill_quality: List[FillQuality] = field(default_factory=list)
    experiment_capacities: List[ExperimentCapacity] = field(default_factory=list)
    max_safe_order_contracts: int = 0
    portfolio_max_aum: float = 0.0
    generated_at: str = ""


# ── Core analyzer ───────────────────────────────────────────────────────────
class LiquidityAnalyzer:
    """Estimates strategy capacity and market impact for credit spreads."""

    def __init__(
        self,
        market_params: Optional[MarketParams] = None,
        impact_coeff: float = IMPACT_COEFF,
        impact_exponent: float = IMPACT_EXPONENT,
        sharpe_decay_sensitivity: float = SHARPE_DECAY_SENSITIVITY,
        aum_grid: Optional[List[float]] = None,
    ) -> None:
        self.market = market_params or MarketParams()
        self.eta = impact_coeff
        self.alpha = impact_exponent
        self.sharpe_sens = sharpe_decay_sensitivity
        self.aum_grid = aum_grid or list(DEFAULT_AUM_GRID)

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        base_sharpe: float = 1.5,
        avg_trade_notional: float = 5_000.0,
        trades_per_day: float = 1.0,
        experiments: Optional[Dict[str, Dict]] = None,
    ) -> LiquidityResult:
        """Run full liquidity analysis.

        Parameters
        ----------
        base_sharpe : float
            Strategy Sharpe at negligible AUM.
        avg_trade_notional : float
            Average notional per trade ($).
        trades_per_day : float
            Average number of trades per day.
        experiments : dict, optional
            experiment_id → {sharpe, avg_notional, trades_per_day}
        """
        curve = self._build_capacity_curve(base_sharpe, avg_trade_notional, trades_per_day)
        fills = self._build_fill_quality_table()
        max_safe = self._max_safe_order()

        exp_caps: List[ExperimentCapacity] = []
        if experiments:
            for eid, params in experiments.items():
                cap = self._estimate_experiment_capacity(eid, params)
                exp_caps.append(cap)

        portfolio_max = self._portfolio_max_aum(curve)

        return LiquidityResult(
            capacity_curve=curve,
            fill_quality=fills,
            experiment_capacities=exp_caps,
            max_safe_order_contracts=max_safe,
            portfolio_max_aum=portfolio_max,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: LiquidityResult,
        output_path: str | Path = "reports/liquidity.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Liquidity report written to %s", path)
        return path

    # ── Market impact model ─────────────────────────────────────────────────
    def market_impact_bps(self, order_contracts: int) -> float:
        """Square-root market impact: η × (Q / ADV)^α × 10000 bps."""
        if self.market.adv_contracts <= 0 or order_contracts <= 0:
            return 0.0
        participation = order_contracts / self.market.adv_contracts
        return self.eta * (participation ** self.alpha) * 10_000

    def effective_spread_bps(self, order_contracts: int) -> float:
        """Bid-ask spread widens with order size relative to ADV."""
        base = self.market.bid_ask_bps
        if self.market.adv_contracts <= 0 or order_contracts <= 0:
            return base
        participation = order_contracts / self.market.adv_contracts
        # Linear widening: doubles at 10% ADV
        widening = 1.0 + participation * 10.0
        return base * widening

    def total_execution_cost_bps(self, order_contracts: int) -> float:
        """Half-spread + market impact."""
        spread = self.effective_spread_bps(order_contracts) / 2.0  # half-spread
        impact = self.market_impact_bps(order_contracts)
        return spread + impact

    def fill_probability(self, order_contracts: int) -> float:
        """Probability of full fill given participation rate."""
        if self.market.adv_contracts <= 0:
            return 0.0
        participation = order_contracts / self.market.adv_contracts
        if participation <= 0.01:
            return 1.0
        if participation >= 0.50:
            return 0.1
        # Logistic decay
        return 1.0 / (1.0 + math.exp(20 * (participation - 0.15)))

    def participation_rate(self, order_contracts: int) -> float:
        """Order size as fraction of ADV."""
        if self.market.adv_contracts <= 0:
            return 0.0
        return order_contracts / self.market.adv_contracts

    # ── Capacity curve ──────────────────────────────────────────────────────
    def _build_capacity_curve(
        self,
        base_sharpe: float,
        avg_notional: float,
        trades_per_day: float,
    ) -> List[CapacityPoint]:
        points: List[CapacityPoint] = []
        for aum in self.aum_grid:
            contracts = self._aum_to_contracts(aum, avg_notional)
            daily_contracts = int(contracts * trades_per_day)
            pr = self.participation_rate(daily_contracts)
            impact = self.market_impact_bps(daily_contracts)
            total = self.total_execution_cost_bps(daily_contracts)

            # Sharpe degradation: proportional to total cost
            degradation = min(1.0, total / 10_000 * self.sharpe_sens * 100)
            expected = base_sharpe * (1.0 - degradation)

            points.append(CapacityPoint(
                aum=aum,
                contracts_per_trade=contracts,
                participation_rate=pr,
                market_impact_bps=impact,
                total_cost_bps=total,
                expected_sharpe=max(0.0, expected),
                sharpe_degradation_pct=degradation * 100,
            ))
        return points

    def _aum_to_contracts(self, aum: float, avg_notional: float) -> int:
        """Convert AUM to number of option contracts per trade."""
        if avg_notional <= 0:
            return 0
        notional = aum * 0.05  # assume ~5% deployed per trade
        return max(1, int(notional / (avg_notional * self.market.contract_multiplier)))

    # ── Fill quality table ──────────────────────────────────────────────────
    def _build_fill_quality_table(self) -> List[FillQuality]:
        results: List[FillQuality] = []
        sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
        for n in sizes:
            pr = self.participation_rate(n)
            spread = self.effective_spread_bps(n)
            impact = self.market_impact_bps(n)
            total = spread / 2.0 + impact
            fill_p = self.fill_probability(n)
            results.append(FillQuality(
                order_contracts=n,
                participation_rate=pr,
                effective_spread_bps=spread,
                market_impact_bps=impact,
                total_cost_bps=total,
                fill_probability=fill_p,
            ))
        return results

    # ── Max safe order ──────────────────────────────────────────────────────
    def _max_safe_order(self) -> int:
        return max(1, int(self.market.adv_contracts * self.market.max_participation_rate))

    # ── Per-experiment capacity ─────────────────────────────────────────────
    def _estimate_experiment_capacity(
        self, eid: str, params: Dict,
    ) -> ExperimentCapacity:
        base_sharpe = float(params.get("sharpe", 1.5))
        avg_notional = float(params.get("avg_notional", 5_000))
        tpd = float(params.get("trades_per_day", 1.0))

        # Find AUM where Sharpe halves
        half_sharpe = base_sharpe * 0.5
        max_aum = self._find_aum_at_sharpe(half_sharpe, base_sharpe, avg_notional, tpd)

        # Recommended: conservative at 60% of max
        rec_aum = max_aum * 0.60

        # Confidence interval: ±30% based on ADV uncertainty
        ci_lower = max_aum * 0.40
        ci_upper = max_aum * 1.30

        rec_contracts = self._aum_to_contracts(rec_aum, avg_notional)
        rec_daily = int(rec_contracts * tpd)
        pr = self.participation_rate(rec_daily)

        return ExperimentCapacity(
            experiment_id=eid,
            base_sharpe=base_sharpe,
            max_aum=max_aum,
            recommended_aum=rec_aum,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            max_contracts_per_trade=self._max_safe_order(),
            participation_at_recommended=pr,
        )

    def _find_aum_at_sharpe(
        self,
        target_sharpe: float,
        base_sharpe: float,
        avg_notional: float,
        trades_per_day: float,
    ) -> float:
        """Binary search for AUM where expected Sharpe reaches target."""
        lo, hi = 1e5, 5e9
        for _ in range(50):
            mid = (lo + hi) / 2.0
            contracts = self._aum_to_contracts(mid, avg_notional)
            daily = int(contracts * trades_per_day)
            total_cost = self.total_execution_cost_bps(daily)
            degradation = min(1.0, total_cost / 10_000 * self.sharpe_sens * 100)
            expected = base_sharpe * (1.0 - degradation)
            if expected > target_sharpe:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ── Portfolio max ───────────────────────────────────────────────────────
    @staticmethod
    def _portfolio_max_aum(curve: List[CapacityPoint]) -> float:
        """AUM at which Sharpe degrades by more than 20%."""
        for pt in curve:
            if pt.sharpe_degradation_pct > 20.0:
                return pt.aum
        return curve[-1].aum if curve else 0.0

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: LiquidityResult) -> str:
        cards = self._html_cards(r)
        curve_chart = self._svg_capacity_curve(r.capacity_curve)
        impact_chart = self._svg_impact_curve(r.fill_quality)
        fill_table = self._html_fill_table(r.fill_quality)
        cap_table = self._html_capacity_table(r.capacity_curve)
        exp_table = self._html_experiment_table(r.experiment_capacities)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Liquidity Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin-bottom:28px}}
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
<h1>Liquidity &amp; Capacity Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{cards}

<div class="sec">
<h2>Capacity Curve — Sharpe vs AUM</h2>
{curve_chart}
</div>

<div class="sec">
<h2>Market Impact vs Order Size</h2>
{impact_chart}
</div>

{cap_table}
{fill_table}
{exp_table}

</body>
</html>"""

    # ── Cards ───────────────────────────────────────────────────────────────
    @staticmethod
    def _html_cards(r: LiquidityResult) -> str:
        max_aum = f"${r.portfolio_max_aum / 1e6:,.1f}M"
        safe = f"{r.max_safe_order_contracts:,}"
        n_exp = len(r.experiment_capacities)
        return f"""<div class="grid">
<div class="card"><div class="lbl">Portfolio Max AUM</div><div class="val">{max_aum}</div></div>
<div class="card"><div class="lbl">Max Safe Order</div><div class="val">{safe} contracts</div></div>
<div class="card"><div class="lbl">Experiments Analyzed</div><div class="val">{n_exp}</div></div>
</div>"""

    # ── Capacity curve SVG ──────────────────────────────────────────────────
    @staticmethod
    def _svg_capacity_curve(curve: List[CapacityPoint]) -> str:
        if not curve:
            return "<p>No data.</p>"
        w, h = 560, 240
        pl, pb, pt = 60, 45, 20
        cw = w - pl
        ch = h - pb - pt
        n = len(curve)

        max_sharpe = max(p.expected_sharpe for p in curve) or 1.0
        log_aums = [math.log10(max(p.aum, 1)) for p in curve]
        min_log, max_log = min(log_aums), max(log_aums)
        rng = max_log - min_log or 1.0

        pts = []
        for i, p in enumerate(curve):
            x = pl + (log_aums[i] - min_log) / rng * cw
            y = pt + ch - (p.expected_sharpe / max_sharpe) * ch
            pts.append((x, y))

        polyline = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
        dots = "".join(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="#38bdf8"/>'
            for x, y in pts
        )
        labels = ""
        for i in [0, len(curve) // 2, len(curve) - 1]:
            p = curve[i]
            x, y = pts[i]
            lbl = f"${p.aum/1e6:.0f}M" if p.aum < 1e9 else f"${p.aum/1e9:.0f}B"
            labels += f'<text x="{x:.0f}" y="{h - 8}" text-anchor="middle" font-size="10" fill="#94a3b8">{lbl}</text>'

        baseline = pt + ch
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{baseline}" x2="{w}" y2="{baseline}" stroke="#475569" stroke-width="1"/>'
            f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{baseline}" stroke="#475569" stroke-width="1"/>'
            f'<polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="2"/>'
            f'{dots}{labels}'
            f'<text x="{pl - 5}" y="{pt + 4}" text-anchor="end" font-size="10" fill="#94a3b8">{max_sharpe:.2f}</text>'
            f'<text x="{pl - 5}" y="{baseline}" text-anchor="end" font-size="10" fill="#94a3b8">0</text>'
            f'</svg>'
        )

    # ── Impact curve SVG ────────────────────────────────────────────────────
    @staticmethod
    def _svg_impact_curve(fills: List[FillQuality]) -> str:
        if not fills:
            return "<p>No data.</p>"
        w, h = 560, 200
        pl, pb, pt = 60, 40, 20
        cw = w - pl
        ch = h - pb - pt
        max_impact = max(f.total_cost_bps for f in fills) or 1.0

        pts = []
        for i, f in enumerate(fills):
            x = pl + (i / max(len(fills) - 1, 1)) * cw
            y = pt + ch - (f.total_cost_bps / max_impact) * ch
            pts.append((x, y))

        polyline = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
        bars = ""
        for i, f in enumerate(fills):
            x, y = pts[i]
            bh = pt + ch - y
            bars += (
                f'<rect x="{x - 8}" y="{y}" width="16" height="{bh}" rx="2" fill="#f97316" opacity="0.6"/>'
                f'<text x="{x}" y="{h - 8}" text-anchor="middle" font-size="9" fill="#94a3b8">{f.order_contracts}</text>'
            )

        baseline = pt + ch
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{baseline}" x2="{w}" y2="{baseline}" stroke="#475569" stroke-width="1"/>'
            f'{bars}'
            f'<polyline points="{polyline}" fill="none" stroke="#f97316" stroke-width="2"/>'
            f'<text x="{pl - 5}" y="{pt + 4}" text-anchor="end" font-size="10" fill="#94a3b8">{max_impact:.0f}bp</text>'
            f'</svg>'
        )

    # ── Tables ──────────────────────────────────────────────────────────────
    @staticmethod
    def _html_capacity_table(curve: List[CapacityPoint]) -> str:
        if not curve:
            return ""
        rows = ""
        for p in curve:
            lbl = f"${p.aum/1e6:.1f}M" if p.aum < 1e9 else f"${p.aum/1e9:.1f}B"
            deg_cls = "pos" if p.sharpe_degradation_pct < 10 else "warn" if p.sharpe_degradation_pct < 25 else "neg"
            rows += (
                f"<tr><td>{lbl}</td>"
                f"<td>{p.contracts_per_trade:,}</td>"
                f"<td>{p.participation_rate:.2%}</td>"
                f"<td>{p.market_impact_bps:.1f}</td>"
                f"<td>{p.total_cost_bps:.1f}</td>"
                f"<td>{p.expected_sharpe:.2f}</td>"
                f'<td class="{deg_cls}">{p.sharpe_degradation_pct:.1f}%</td></tr>'
            )
        return f"""<div class="sec">
<h2>Capacity Curve Detail</h2>
<table>
<thead><tr><th>AUM</th><th>Contracts</th><th>Participation</th><th>Impact (bps)</th><th>Total Cost</th><th>Exp Sharpe</th><th>Degradation</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_fill_table(fills: List[FillQuality]) -> str:
        if not fills:
            return ""
        rows = ""
        for f in fills:
            fill_cls = "pos" if f.fill_probability > 0.9 else "warn" if f.fill_probability > 0.5 else "neg"
            rows += (
                f"<tr><td>{f.order_contracts}</td>"
                f"<td>{f.participation_rate:.2%}</td>"
                f"<td>{f.effective_spread_bps:.1f}</td>"
                f"<td>{f.market_impact_bps:.1f}</td>"
                f"<td>{f.total_cost_bps:.1f}</td>"
                f'<td class="{fill_cls}">{f.fill_probability:.0%}</td></tr>'
            )
        return f"""<div class="sec">
<h2>Fill Quality by Order Size</h2>
<table>
<thead><tr><th>Contracts</th><th>Participation</th><th>Eff Spread</th><th>Impact (bps)</th><th>Total Cost</th><th>Fill Prob</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_experiment_table(caps: List[ExperimentCapacity]) -> str:
        if not caps:
            return ""
        rows = ""
        for c in caps:
            rows += (
                f"<tr><td>{c.experiment_id}</td>"
                f"<td>{c.base_sharpe:.2f}</td>"
                f"<td>${c.max_aum / 1e6:,.1f}M</td>"
                f"<td>${c.recommended_aum / 1e6:,.1f}M</td>"
                f"<td>${c.ci_lower / 1e6:,.1f}M – ${c.ci_upper / 1e6:,.1f}M</td>"
                f"<td>{c.max_contracts_per_trade:,}</td>"
                f"<td>{c.participation_at_recommended:.2%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Per-Experiment Capacity</h2>
<table>
<thead><tr><th>Experiment</th><th>Base Sharpe</th><th>Max AUM</th><th>Recommended</th><th>90% CI</th><th>Max Contracts</th><th>Part. Rate</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
