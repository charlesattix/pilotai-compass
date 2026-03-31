"""compass/overnight_risk.py — Overnight risk assessment for morning review.

Computes overnight gap risk, earnings exposure, expiry risk by DTE bucket,
sector concentration (Herfindahl), correlation exposure, max-loss scenarios
(delta-normal 99th percentile), margin stress buffer, position-level Greeks
summary, and actionable hedging recommendations.

Usage::

    from compass.overnight_risk import OvernightRiskReport

    report = OvernightRiskReport(
        positions=[
            {"symbol": "SPY", "quantity": -10, "delta": -0.30, "gamma": 0.02,
             "vega": 0.15, "theta": -0.05, "dte": 14, "sector": "ETF",
             "entry_price": 2.50},
        ],
        portfolio_value=100_000.0,
        vix=18.5,
    )
    result = report.generate()
    html = report.generate_report()
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

# Historical overnight gap parameters (empirical S&P 500 overnight returns)
# Mean and std of overnight log-returns used for parametric gap model.
OVERNIGHT_GAP_MEAN: float = 0.0002
OVERNIGHT_GAP_STD: float = 0.0055

# DTE bucket boundaries
DTE_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 7, "0-7"),
    (7, 30, "7-30"),
    (30, 60, "30-60"),
    (60, 999999, "60+"),
]

# VIX thresholds for hedge recommendations
VIX_LOW: float = 15.0
VIX_HIGH: float = 25.0

# Stress margin multiplier applied on top of current margin for buffer calc
STRESS_MARGIN_MULTIPLIER: float = 1.5

# Z-scores
Z_95: float = 1.6449
Z_99: float = 2.3263


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class GapRiskResult:
    """Overnight gap risk estimate."""
    gap_std: float
    pct_95_loss: float
    pct_99_loss: float
    dollar_95_loss: float
    dollar_99_loss: float
    portfolio_pct_95: float


@dataclass
class EarningsExposure:
    """Earnings events within the look-ahead window."""
    symbol: str
    earnings_date: str
    position_delta: float
    position_vega: float
    notional_at_risk: float


@dataclass
class DTEBucket:
    """Expiry risk breakdown for a DTE range."""
    label: str
    count: int
    total_theta: float
    total_delta: float
    total_notional: float


@dataclass
class SectorConcentration:
    """Sector concentration metrics."""
    herfindahl: float
    max_sector: str
    max_sector_weight: float
    sector_weights: Dict[str, float]


@dataclass
class CorrelationExposure:
    """Portfolio correlation to SPY and VIX."""
    spy_beta: float
    vix_beta: float
    net_spy_dollar_delta: float
    net_vix_dollar_delta: float


@dataclass
class MaxLossScenario:
    """Delta-normal max loss at 99th percentile."""
    var_99: float
    var_95: float
    portfolio_pct: float
    dominant_risk_factor: str


@dataclass
class MarginBuffer:
    """Margin stress analysis."""
    current_margin: float
    stress_margin: float
    margin_buffer: float
    buffer_pct: float
    is_adequate: bool


@dataclass
class PositionGreeksSummary:
    """Per-position Greeks row for the summary table."""
    symbol: str
    quantity: int
    delta: float
    gamma: float
    vega: float
    theta: float
    dte: int
    sector: str
    entry_price: float
    dollar_delta: float
    dollar_gamma: float
    dollar_vega: float
    dollar_theta: float


@dataclass
class HedgeRecommendation:
    """A single hedging action recommendation."""
    action: str
    reason: str
    priority: str  # "high", "medium", "low"
    instrument: str


@dataclass
class OvernightResult:
    """Full overnight risk report."""
    timestamp: str
    portfolio_value: float
    vix: float
    gap_risk: GapRiskResult
    earnings_exposures: List[EarningsExposure]
    dte_buckets: List[DTEBucket]
    sector_concentration: SectorConcentration
    correlation_exposure: CorrelationExposure
    max_loss: MaxLossScenario
    margin_buffer: MarginBuffer
    greeks_summary: List[PositionGreeksSummary]
    hedge_recommendations: List[HedgeRecommendation]
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float


# ── Core class ──────────────────────────────────────────────────────────────


class OvernightRiskReport:
    """Generate an overnight risk assessment suitable for a 5-minute morning review."""

    def __init__(
        self,
        positions: List[Dict[str, Any]],
        portfolio_value: float,
        vix: float,
        earnings_calendar: Optional[List[Dict[str, str]]] = None,
        spy_beta: float = 1.0,
        vix_beta: float = -0.10,
        current_margin: float = 0.0,
        as_of: Optional[date] = None,
    ) -> None:
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.vix = vix
        self.earnings_calendar = earnings_calendar or []
        self.spy_beta = spy_beta
        self.vix_beta = vix_beta
        self.current_margin = current_margin
        self.as_of = as_of or date.today()

    # ── Individual computations ─────────────────────────────────────────

    def compute_gap_risk(self) -> GapRiskResult:
        """Parametric overnight gap risk using historical gap distribution."""
        if not self.positions:
            return GapRiskResult(
                gap_std=OVERNIGHT_GAP_STD,
                pct_95_loss=0.0,
                pct_99_loss=0.0,
                dollar_95_loss=0.0,
                dollar_99_loss=0.0,
                portfolio_pct_95=0.0,
            )

        # Portfolio-weighted dollar delta exposure
        total_dollar_delta = sum(
            p["delta"] * p["quantity"] * p["entry_price"] * 100
            for p in self.positions
        )

        # Scale gap std by VIX relative to long-run mean (~18)
        vix_scalar = max(self.vix / 18.0, 0.5)
        adj_std = OVERNIGHT_GAP_STD * vix_scalar

        pct_95_loss = abs(total_dollar_delta) * Z_95 * adj_std
        pct_99_loss = abs(total_dollar_delta) * Z_99 * adj_std

        portfolio_pct_95 = (
            pct_95_loss / self.portfolio_value * 100
            if self.portfolio_value > 0
            else 0.0
        )

        return GapRiskResult(
            gap_std=adj_std,
            pct_95_loss=round(pct_95_loss, 2),
            pct_99_loss=round(pct_99_loss, 2),
            dollar_95_loss=round(pct_95_loss, 2),
            dollar_99_loss=round(pct_99_loss, 2),
            portfolio_pct_95=round(portfolio_pct_95, 4),
        )

    def compute_earnings_exposure(self) -> List[EarningsExposure]:
        """Identify positions with earnings events in the next 5 days."""
        if not self.earnings_calendar:
            return []

        window_end = self.as_of + timedelta(days=5)
        exposures: List[EarningsExposure] = []

        # Build a map of symbol -> position for quick lookup
        pos_map: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.positions:
            pos_map.setdefault(p["symbol"], []).append(p)

        for event in self.earnings_calendar:
            sym = event["symbol"]
            edate_raw = event["date"]
            if isinstance(edate_raw, str):
                edate = date.fromisoformat(edate_raw)
            elif isinstance(edate_raw, date):
                edate = edate_raw
            else:
                continue

            if self.as_of <= edate <= window_end and sym in pos_map:
                for p in pos_map[sym]:
                    notional = abs(p["quantity"] * p["entry_price"] * 100)
                    exposures.append(
                        EarningsExposure(
                            symbol=sym,
                            earnings_date=str(edate),
                            position_delta=p["delta"] * p["quantity"],
                            position_vega=p["vega"] * p["quantity"],
                            notional_at_risk=round(notional, 2),
                        )
                    )

        return exposures

    def compute_dte_buckets(self) -> List[DTEBucket]:
        """Break positions into DTE buckets for expiry risk analysis."""
        buckets: Dict[str, DTEBucket] = {}
        for lo, hi, label in DTE_BUCKETS:
            buckets[label] = DTEBucket(
                label=label, count=0, total_theta=0.0,
                total_delta=0.0, total_notional=0.0,
            )

        for p in self.positions:
            dte = p["dte"]
            for lo, hi, label in DTE_BUCKETS:
                if lo <= dte < hi:
                    b = buckets[label]
                    b.count += abs(p["quantity"])
                    b.total_theta += p["theta"] * p["quantity"]
                    b.total_delta += p["delta"] * p["quantity"]
                    b.total_notional += abs(p["quantity"] * p["entry_price"] * 100)
                    break

        return [
            DTEBucket(
                label=b.label,
                count=b.count,
                total_theta=round(b.total_theta, 4),
                total_delta=round(b.total_delta, 4),
                total_notional=round(b.total_notional, 2),
            )
            for b in buckets.values()
        ]

    def compute_sector_concentration(self) -> SectorConcentration:
        """Herfindahl index and max sector weight."""
        if not self.positions:
            return SectorConcentration(
                herfindahl=0.0,
                max_sector="N/A",
                max_sector_weight=0.0,
                sector_weights={},
            )

        sector_notional: Dict[str, float] = {}
        total_notional = 0.0
        for p in self.positions:
            notional = abs(p["quantity"] * p["entry_price"] * 100)
            sector = p.get("sector", "Unknown")
            sector_notional[sector] = sector_notional.get(sector, 0.0) + notional
            total_notional += notional

        if total_notional == 0:
            return SectorConcentration(
                herfindahl=0.0,
                max_sector="N/A",
                max_sector_weight=0.0,
                sector_weights={},
            )

        weights = {s: v / total_notional for s, v in sector_notional.items()}
        hhi = sum(w ** 2 for w in weights.values())
        max_sector = max(weights, key=weights.get)  # type: ignore[arg-type]
        max_weight = weights[max_sector]

        return SectorConcentration(
            herfindahl=round(hhi, 6),
            max_sector=max_sector,
            max_sector_weight=round(max_weight, 4),
            sector_weights={s: round(w, 4) for s, w in weights.items()},
        )

    def compute_correlation_exposure(self) -> CorrelationExposure:
        """Portfolio correlation exposure to SPY and VIX."""
        net_dollar_delta = sum(
            p["delta"] * p["quantity"] * p["entry_price"] * 100
            for p in self.positions
        )

        spy_dollar = net_dollar_delta * self.spy_beta
        vix_dollar = net_dollar_delta * self.vix_beta

        return CorrelationExposure(
            spy_beta=self.spy_beta,
            vix_beta=self.vix_beta,
            net_spy_dollar_delta=round(spy_dollar, 2),
            net_vix_dollar_delta=round(vix_dollar, 2),
        )

    def compute_max_loss(self) -> MaxLossScenario:
        """Delta-normal VaR at 99th percentile."""
        if not self.positions:
            return MaxLossScenario(
                var_99=0.0,
                var_95=0.0,
                portfolio_pct=0.0,
                dominant_risk_factor="none",
            )

        # Dollar Greeks
        total_dollar_delta = sum(
            p["delta"] * p["quantity"] * p["entry_price"] * 100
            for p in self.positions
        )
        total_dollar_vega = sum(
            p["vega"] * p["quantity"] * p["entry_price"] * 100
            for p in self.positions
        )

        # Annualised vol of underlying scaled to daily
        daily_vol = self.vix / 100.0 / math.sqrt(252)

        # Delta component
        delta_loss = abs(total_dollar_delta) * daily_vol
        # Vega component: 1-point vol move
        vega_loss = abs(total_dollar_vega) * 0.01

        # Combined (assume ~0.5 correlation between delta and vega risk)
        combined_std = math.sqrt(delta_loss ** 2 + vega_loss ** 2 + 2 * 0.5 * delta_loss * vega_loss)

        var_99 = Z_99 * combined_std
        var_95 = Z_95 * combined_std

        portfolio_pct = (
            var_99 / self.portfolio_value * 100
            if self.portfolio_value > 0
            else 0.0
        )

        dominant = "delta" if delta_loss >= vega_loss else "vega"

        return MaxLossScenario(
            var_99=round(var_99, 2),
            var_95=round(var_95, 2),
            portfolio_pct=round(portfolio_pct, 4),
            dominant_risk_factor=dominant,
        )

    def compute_margin_buffer(self) -> MarginBuffer:
        """Stress margin requirement and buffer analysis."""
        if self.current_margin <= 0:
            return MarginBuffer(
                current_margin=0.0,
                stress_margin=0.0,
                margin_buffer=0.0,
                buffer_pct=0.0,
                is_adequate=True,
            )

        # Stress margin: inflate by VIX-scaled multiplier
        vix_factor = max(self.vix / 18.0, 1.0)
        stress_mult = STRESS_MARGIN_MULTIPLIER * vix_factor
        stress_margin = self.current_margin * stress_mult

        buffer = self.portfolio_value - stress_margin
        buffer_pct = (
            buffer / self.portfolio_value * 100
            if self.portfolio_value > 0
            else 0.0
        )

        return MarginBuffer(
            current_margin=round(self.current_margin, 2),
            stress_margin=round(stress_margin, 2),
            margin_buffer=round(buffer, 2),
            buffer_pct=round(buffer_pct, 4),
            is_adequate=buffer > 0,
        )

    def compute_greeks_summary(self) -> List[PositionGreeksSummary]:
        """Per-position Greeks summary table."""
        summaries: List[PositionGreeksSummary] = []
        for p in self.positions:
            qty = p["quantity"]
            price = p["entry_price"]
            multiplier = price * 100
            summaries.append(
                PositionGreeksSummary(
                    symbol=p["symbol"],
                    quantity=qty,
                    delta=p["delta"],
                    gamma=p["gamma"],
                    vega=p["vega"],
                    theta=p["theta"],
                    dte=p["dte"],
                    sector=p.get("sector", "Unknown"),
                    entry_price=price,
                    dollar_delta=round(p["delta"] * qty * multiplier, 2),
                    dollar_gamma=round(p["gamma"] * qty * multiplier, 2),
                    dollar_vega=round(p["vega"] * qty * multiplier, 2),
                    dollar_theta=round(p["theta"] * qty * multiplier, 2),
                )
            )
        return summaries

    def compute_hedge_recommendations(self) -> List[HedgeRecommendation]:
        """Generate hedging recommendations based on exposure and VIX level."""
        recs: List[HedgeRecommendation] = []
        if not self.positions:
            return recs

        net_delta = sum(p["delta"] * p["quantity"] for p in self.positions)
        net_vega = sum(p["vega"] * p["quantity"] for p in self.positions)
        net_gamma = sum(p["gamma"] * p["quantity"] for p in self.positions)

        # Delta exposure check
        if abs(net_delta) > 0.5:
            direction = "long" if net_delta > 0 else "short"
            recs.append(
                HedgeRecommendation(
                    action=f"Reduce {direction} delta exposure ({net_delta:+.2f})",
                    reason=f"Net delta of {net_delta:+.2f} exceeds +/-0.50 threshold",
                    priority="high" if abs(net_delta) > 1.0 else "medium",
                    instrument="SPY shares or delta-neutral spreads",
                )
            )

        # Vega exposure in high-VIX regime
        if self.vix > VIX_HIGH and net_vega > 0:
            recs.append(
                HedgeRecommendation(
                    action="Reduce long vega exposure in elevated VIX environment",
                    reason=f"VIX at {self.vix:.1f} (>{VIX_HIGH}) with positive net vega ({net_vega:+.2f})",
                    priority="high",
                    instrument="Short VIX calls or sell premium",
                )
            )

        # Vega exposure in low-VIX regime
        if self.vix < VIX_LOW and net_vega < -0.5:
            recs.append(
                HedgeRecommendation(
                    action="Consider adding long vega as VIX is suppressed",
                    reason=f"VIX at {self.vix:.1f} (<{VIX_LOW}) with short vega ({net_vega:+.2f})",
                    priority="medium",
                    instrument="Long puts or VIX calls for tail protection",
                )
            )

        # High VIX general caution
        if self.vix > VIX_HIGH:
            recs.append(
                HedgeRecommendation(
                    action="Consider reducing overall position size",
                    reason=f"VIX elevated at {self.vix:.1f} — overnight gaps more likely",
                    priority="high",
                    instrument="Reduce notional or add protective puts",
                )
            )

        # Negative gamma warning
        if net_gamma < -0.1:
            recs.append(
                HedgeRecommendation(
                    action="Negative gamma exposure — hedge with long options",
                    reason=f"Net gamma {net_gamma:+.2f} creates convex losses on large moves",
                    priority="medium" if net_gamma > -0.5 else "high",
                    instrument="Long straddles or strangles near current price",
                )
            )

        # DTE proximity warning
        short_dte_positions = [p for p in self.positions if p["dte"] <= 7]
        if short_dte_positions:
            total_short_dte = sum(abs(p["quantity"]) for p in short_dte_positions)
            recs.append(
                HedgeRecommendation(
                    action=f"Monitor {total_short_dte} contracts expiring within 7 days",
                    reason="Gamma risk accelerates near expiry",
                    priority="medium",
                    instrument="Roll or close short-DTE positions",
                )
            )

        return recs

    # ── Aggregate generation ────────────────────────────────────────────

    def generate(self) -> OvernightResult:
        """Compute all risk sections and return a full OvernightResult."""
        net_delta = sum(p["delta"] * p["quantity"] for p in self.positions)
        net_gamma = sum(p["gamma"] * p["quantity"] for p in self.positions)
        net_vega = sum(p["vega"] * p["quantity"] for p in self.positions)
        net_theta = sum(p["theta"] * p["quantity"] for p in self.positions)

        return OvernightResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            portfolio_value=self.portfolio_value,
            vix=self.vix,
            gap_risk=self.compute_gap_risk(),
            earnings_exposures=self.compute_earnings_exposure(),
            dte_buckets=self.compute_dte_buckets(),
            sector_concentration=self.compute_sector_concentration(),
            correlation_exposure=self.compute_correlation_exposure(),
            max_loss=self.compute_max_loss(),
            margin_buffer=self.compute_margin_buffer(),
            greeks_summary=self.compute_greeks_summary(),
            hedge_recommendations=self.compute_hedge_recommendations(),
            net_delta=round(net_delta, 4),
            net_gamma=round(net_gamma, 4),
            net_vega=round(net_vega, 4),
            net_theta=round(net_theta, 4),
        )

    # ── HTML report ─────────────────────────────────────────────────────

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate an HTML report suitable for a 5-minute morning review.

        Returns the HTML string. If *output_path* is given, also writes to disk.
        """
        result = self.generate()
        html = self._render_html(result)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("Overnight risk report written to %s", output_path)
        return html

    def _render_html(self, r: OvernightResult) -> str:
        """Build the HTML document."""
        cards = self._render_summary_cards(r)
        gap_section = self._render_gap_risk(r.gap_risk)
        earnings_section = self._render_earnings(r.earnings_exposures)
        dte_section = self._render_dte_buckets(r.dte_buckets)
        sector_section = self._render_sector(r.sector_concentration)
        maxloss_section = self._render_max_loss(r.max_loss)
        margin_section = self._render_margin(r.margin_buffer)
        greeks_table = self._render_greeks_table(r.greeks_summary)
        recs_section = self._render_recommendations(r.hedge_recommendations)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overnight Risk Report</title>
<style>
  :root {{ --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
           --text: #e0e0e0; --accent: #4fc3f7; --red: #ef5350;
           --green: #66bb6a; --yellow: #ffa726; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif;
          background: var(--bg); color: var(--text); padding: 20px; }}
  h1 {{ color: var(--accent); margin-bottom: 4px; font-size: 1.5rem; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }}
  .card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase;
                  letter-spacing: 0.5px; }}
  .card .value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
  .card .value.red {{ color: var(--red); }}
  .card .value.green {{ color: var(--green); }}
  .card .value.yellow {{ color: var(--yellow); }}
  .section {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .section h2 {{ font-size: 1rem; color: var(--accent); margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--border); }}
  th {{ color: #888; text-transform: uppercase; font-size: 0.7rem;
       letter-spacing: 0.5px; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .rec {{ padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
          border-left: 4px solid; }}
  .rec.high {{ background: rgba(239,83,80,0.1); border-color: var(--red); }}
  .rec.medium {{ background: rgba(255,167,38,0.1); border-color: var(--yellow); }}
  .rec.low {{ background: rgba(102,187,106,0.1); border-color: var(--green); }}
  .rec .action {{ font-weight: 600; }}
  .rec .reason {{ font-size: 0.8rem; color: #aaa; margin-top: 2px; }}
  .rec .instrument {{ font-size: 0.75rem; color: #666; margin-top: 2px; }}
</style>
</head>
<body>
<h1>Overnight Risk Report</h1>
<div class="subtitle">Generated {r.timestamp} | VIX {r.vix:.1f} | Portfolio ${r.portfolio_value:,.0f}</div>

{cards}
{gap_section}
{earnings_section}
{dte_section}
{sector_section}
{maxloss_section}
{margin_section}
{greeks_table}
{recs_section}

</body>
</html>"""

    def _render_summary_cards(self, r: OvernightResult) -> str:
        vix_class = "red" if r.vix > VIX_HIGH else ("yellow" if r.vix > VIX_LOW else "green")
        gap_class = "red" if r.gap_risk.portfolio_pct_95 > 2 else ("yellow" if r.gap_risk.portfolio_pct_95 > 1 else "green")
        margin_class = "green" if r.margin_buffer.is_adequate else "red"

        return f"""<div class="cards">
  <div class="card"><div class="label">VIX</div><div class="value {vix_class}">{r.vix:.1f}</div></div>
  <div class="card"><div class="label">Net Delta</div><div class="value">{r.net_delta:+.2f}</div></div>
  <div class="card"><div class="label">Net Vega</div><div class="value">{r.net_vega:+.2f}</div></div>
  <div class="card"><div class="label">Net Theta</div><div class="value">{r.net_theta:+.2f}</div></div>
  <div class="card"><div class="label">Gap Risk (95%)</div><div class="value {gap_class}">${r.gap_risk.dollar_95_loss:,.0f}</div></div>
  <div class="card"><div class="label">Max Loss (99%)</div><div class="value red">${r.max_loss.var_99:,.0f}</div></div>
  <div class="card"><div class="label">Margin Buffer</div><div class="value {margin_class}">{r.margin_buffer.buffer_pct:.1f}%</div></div>
  <div class="card"><div class="label">Earnings Exposed</div><div class="value {'red' if r.earnings_exposures else 'green'}">{len(r.earnings_exposures)}</div></div>
</div>"""

    def _render_gap_risk(self, g: GapRiskResult) -> str:
        return f"""<div class="section">
  <h2>Overnight Gap Risk</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Adjusted Gap Std</td><td>{g.gap_std:.4f}</td></tr>
    <tr><td>95th Percentile Loss</td><td>${g.dollar_95_loss:,.2f}</td></tr>
    <tr><td>99th Percentile Loss</td><td>${g.dollar_99_loss:,.2f}</td></tr>
    <tr><td>Portfolio % (95th)</td><td>{g.portfolio_pct_95:.2f}%</td></tr>
  </table>
</div>"""

    def _render_earnings(self, exposures: List[EarningsExposure]) -> str:
        if not exposures:
            return """<div class="section"><h2>Earnings Exposure (Next 5 Days)</h2><p>No earnings events detected.</p></div>"""
        rows = "".join(
            f"<tr><td>{e.symbol}</td><td>{e.earnings_date}</td>"
            f"<td>{e.position_delta:+.2f}</td><td>{e.position_vega:+.2f}</td>"
            f"<td>${e.notional_at_risk:,.0f}</td></tr>"
            for e in exposures
        )
        return f"""<div class="section">
  <h2>Earnings Exposure (Next 5 Days)</h2>
  <table>
    <tr><th>Symbol</th><th>Date</th><th>Delta</th><th>Vega</th><th>Notional</th></tr>
    {rows}
  </table>
</div>"""

    def _render_dte_buckets(self, buckets: List[DTEBucket]) -> str:
        rows = "".join(
            f"<tr><td>{b.label}</td><td>{b.count}</td>"
            f"<td>{b.total_delta:+.4f}</td><td>{b.total_theta:+.4f}</td>"
            f"<td>${b.total_notional:,.0f}</td></tr>"
            for b in buckets
        )
        return f"""<div class="section">
  <h2>Expiry Risk by DTE</h2>
  <table>
    <tr><th>DTE Bucket</th><th>Contracts</th><th>Delta</th><th>Theta</th><th>Notional</th></tr>
    {rows}
  </table>
</div>"""

    def _render_sector(self, s: SectorConcentration) -> str:
        rows = "".join(
            f"<tr><td>{sec}</td><td>{w * 100:.1f}%</td></tr>"
            for sec, w in sorted(s.sector_weights.items(), key=lambda x: -x[1])
        )
        return f"""<div class="section">
  <h2>Sector Concentration</h2>
  <p>Herfindahl Index: {s.herfindahl:.4f} | Max Sector: {s.max_sector} ({s.max_sector_weight * 100:.1f}%)</p>
  <table>
    <tr><th>Sector</th><th>Weight</th></tr>
    {rows}
  </table>
</div>"""

    def _render_max_loss(self, m: MaxLossScenario) -> str:
        return f"""<div class="section">
  <h2>Max Loss Scenario (Delta-Normal)</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>VaR 95%</td><td>${m.var_95:,.2f}</td></tr>
    <tr><td>VaR 99%</td><td>${m.var_99:,.2f}</td></tr>
    <tr><td>Portfolio %</td><td>{m.portfolio_pct:.2f}%</td></tr>
    <tr><td>Dominant Risk Factor</td><td>{m.dominant_risk_factor}</td></tr>
  </table>
</div>"""

    def _render_margin(self, m: MarginBuffer) -> str:
        status = "Adequate" if m.is_adequate else "INSUFFICIENT"
        return f"""<div class="section">
  <h2>Margin Buffer</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Current Margin</td><td>${m.current_margin:,.2f}</td></tr>
    <tr><td>Stress Margin</td><td>${m.stress_margin:,.2f}</td></tr>
    <tr><td>Buffer</td><td>${m.margin_buffer:,.2f}</td></tr>
    <tr><td>Buffer %</td><td>{m.buffer_pct:.1f}%</td></tr>
    <tr><td>Status</td><td>{status}</td></tr>
  </table>
</div>"""

    def _render_greeks_table(self, summaries: List[PositionGreeksSummary]) -> str:
        if not summaries:
            return """<div class="section"><h2>Position Greeks</h2><p>No positions.</p></div>"""
        rows = "".join(
            f"<tr><td>{s.symbol}</td><td>{s.quantity}</td>"
            f"<td>{s.delta:+.4f}</td><td>{s.gamma:+.4f}</td>"
            f"<td>{s.vega:+.4f}</td><td>{s.theta:+.4f}</td>"
            f"<td>{s.dte}</td><td>{s.sector}</td>"
            f"<td>${s.dollar_delta:,.0f}</td><td>${s.dollar_vega:,.0f}</td></tr>"
            for s in summaries
        )
        return f"""<div class="section">
  <h2>Position Greeks</h2>
  <table>
    <tr><th>Symbol</th><th>Qty</th><th>Delta</th><th>Gamma</th><th>Vega</th>
        <th>Theta</th><th>DTE</th><th>Sector</th><th>$Delta</th><th>$Vega</th></tr>
    {rows}
  </table>
</div>"""

    def _render_recommendations(self, recs: List[HedgeRecommendation]) -> str:
        if not recs:
            return """<div class="section"><h2>Hedge Recommendations</h2><p>No actions recommended at this time.</p></div>"""
        items = "".join(
            f'<div class="rec {r.priority}">'
            f'<div class="action">[{r.priority.upper()}] {r.action}</div>'
            f'<div class="reason">{r.reason}</div>'
            f'<div class="instrument">Instrument: {r.instrument}</div></div>'
            for r in recs
        )
        return f"""<div class="section">
  <h2>Hedge Recommendations</h2>
  {items}
</div>"""
