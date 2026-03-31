"""Dynamic hedging engine — computes optimal hedge ratios in real-time for
VIX call overlays, SPY delta hedges, and correlation-based cross-hedges.

Extends (does not replace) the static ``CrisisHedgeController`` from
``compass.crisis_hedge`` by adding:
  1. VIX call overlay sizing (tail risk protection)
  2. SPY delta hedge (neutralise directional exposure)
  3. Correlation-based cross-hedge (when experiments become correlated)
  4. Cost–benefit analysis (hedge cost vs expected DD reduction)
  5. Regime-conditional rules (bull / bear / crash differ)
  6. Hedge P&L tracking (separate hedge returns from alpha returns)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime constants ────────────────────────────────────────────────────────
BULL = "bull"
BEAR = "bear"
HIGH_VOL = "high_vol"
LOW_VOL = "low_vol"
CRASH = "crash"

ALL_REGIMES = [BULL, BEAR, HIGH_VOL, LOW_VOL, CRASH]

# ── Default parameters ──────────────────────────────────────────────────────
DEFAULT_VIX_CALL_BUDGET_PCT = 0.02       # 2% of portfolio for VIX call protection
DEFAULT_DELTA_HEDGE_THRESHOLD = 0.10     # hedge when abs(portfolio delta) > 10%
DEFAULT_CORR_HEDGE_THRESHOLD = 0.70      # cross-hedge when corr > 0.70
DEFAULT_REBALANCE_TOLERANCE = 0.05       # 5% drift before rebalance


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class HedgeConfig:
    """Configuration for the dynamic hedge engine."""
    vix_call_budget_pct: float = DEFAULT_VIX_CALL_BUDGET_PCT
    delta_hedge_threshold: float = DEFAULT_DELTA_HEDGE_THRESHOLD
    corr_hedge_threshold: float = DEFAULT_CORR_HEDGE_THRESHOLD
    rebalance_tolerance: float = DEFAULT_REBALANCE_TOLERANCE
    # Regime-specific VIX call scaling
    regime_vix_scale: Dict[str, float] = field(default_factory=lambda: {
        BULL: 0.5,      # half budget in bull
        BEAR: 1.5,      # 1.5× in bear
        HIGH_VOL: 2.0,  # 2× in high vol
        LOW_VOL: 0.25,  # minimal in low vol
        CRASH: 3.0,     # maximum in crash
    })
    # Regime-specific delta hedge aggressiveness (0=none, 1=full)
    regime_delta_aggression: Dict[str, float] = field(default_factory=lambda: {
        BULL: 0.3,
        BEAR: 0.8,
        HIGH_VOL: 0.9,
        LOW_VOL: 0.2,
        CRASH: 1.0,
    })


@dataclass
class VIXCallOverlay:
    """Sizing for VIX call tail protection."""
    notional_pct: float       # % of portfolio allocated
    n_contracts: int          # number of VIX call contracts
    strike_offset: float      # points above current VIX
    estimated_cost: float     # estimated premium cost
    expected_payoff_at_spike: float  # expected payoff if VIX doubles
    regime_multiplier: float


@dataclass
class DeltaHedge:
    """SPY delta hedge recommendation."""
    portfolio_delta: float    # net portfolio delta exposure
    hedge_delta: float        # delta to offset (negative = short SPY)
    hedge_ratio: float        # fraction of delta neutralised
    spy_shares: int           # shares of SPY to trade
    regime_aggression: float


@dataclass
class CrossHedge:
    """Correlation-based cross-hedge between experiments."""
    exp_a: str
    exp_b: str
    correlation: float
    hedge_action: str         # "reduce_both", "diversify", "none"
    recommended_weight_adj: float  # multiplier for position reduction


@dataclass
class CostBenefit:
    """Cost-benefit analysis of hedging strategy."""
    total_hedge_cost: float         # annualised cost as % of portfolio
    expected_dd_reduction_pct: float  # expected DD reduction
    cost_per_dd_point: float        # cost efficiency
    break_even_dd: float            # DD at which hedge pays for itself
    net_benefit: float              # expected_dd_reduction - cost (positive = good)
    recommendation: str             # "hedge", "partial_hedge", "no_hedge"


@dataclass
class HedgePnL:
    """Track hedge P&L separately from alpha."""
    alpha_pnl: float = 0.0
    vix_call_pnl: float = 0.0
    delta_hedge_pnl: float = 0.0
    cross_hedge_pnl: float = 0.0
    total_hedge_pnl: float = 0.0
    total_pnl: float = 0.0
    hedge_drag_pct: float = 0.0    # hedge cost as fraction of alpha


@dataclass
class HedgeSnapshot:
    """Complete hedge state at a point in time."""
    regime: str
    vix_level: float
    vix_overlay: Optional[VIXCallOverlay] = None
    delta_hedge: Optional[DeltaHedge] = None
    cross_hedges: List[CrossHedge] = field(default_factory=list)
    cost_benefit: Optional[CostBenefit] = None
    hedge_pnl: Optional[HedgePnL] = None
    generated_at: str = ""


@dataclass
class HedgeHistory:
    """Full history of hedge decisions for reporting."""
    snapshots: List[HedgeSnapshot] = field(default_factory=list)
    cumulative_pnl: Optional[HedgePnL] = None
    generated_at: str = ""


# ── Core engine ─────────────────────────────────────────────────────────────
class DynamicHedgeEngine:
    """Computes optimal hedge ratios across VIX overlays, delta hedges,
    and correlation-based cross-hedges."""

    def __init__(self, config: Optional[HedgeConfig] = None) -> None:
        self.config = config or HedgeConfig()
        self._history: List[HedgeSnapshot] = []

    # ── Public API ──────────────────────────────────────────────────────────
    def compute_hedges(
        self,
        portfolio_value: float,
        vix: float,
        regime: str,
        portfolio_delta: float = 0.0,
        spy_price: float = 450.0,
        experiment_returns: Optional[Dict[str, pd.Series]] = None,
        experiment_weights: Optional[Dict[str, float]] = None,
    ) -> HedgeSnapshot:
        """Compute all hedge recommendations for current market state.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio NAV.
        vix : float
            Current VIX level.
        regime : str
            Current market regime (bull/bear/high_vol/low_vol/crash).
        portfolio_delta : float
            Net portfolio delta (positive = long market).
        spy_price : float
            Current SPY price for delta hedge sizing.
        experiment_returns : dict, optional
            experiment_id → return series for cross-hedge analysis.
        experiment_weights : dict, optional
            experiment_id → portfolio weight.
        """
        vix_overlay = self._compute_vix_overlay(portfolio_value, vix, regime)
        delta = self._compute_delta_hedge(portfolio_delta, regime, spy_price)

        cross_hedges: List[CrossHedge] = []
        if experiment_returns and experiment_weights:
            cross_hedges = self._compute_cross_hedges(
                experiment_returns, experiment_weights,
            )

        cost_benefit = self._compute_cost_benefit(
            portfolio_value, vix_overlay, delta, vix,
        )

        snap = HedgeSnapshot(
            regime=regime,
            vix_level=vix,
            vix_overlay=vix_overlay,
            delta_hedge=delta,
            cross_hedges=cross_hedges,
            cost_benefit=cost_benefit,
            generated_at=self._now(),
        )
        self._history.append(snap)
        return snap

    def track_pnl(
        self,
        alpha_pnl: float,
        vix_change: float = 0.0,
        spy_return: float = 0.0,
        snapshot: Optional[HedgeSnapshot] = None,
    ) -> HedgePnL:
        """Estimate hedge P&L given market moves.

        Parameters
        ----------
        alpha_pnl : float
            P&L from credit spread alpha strategy.
        vix_change : float
            Change in VIX level (positive = VIX rose).
        spy_return : float
            SPY return (e.g. -0.02 for -2%).
        snapshot : HedgeSnapshot, optional
            The hedge state; uses latest if None.
        """
        if snapshot is None and self._history:
            snapshot = self._history[-1]

        vix_pnl = 0.0
        delta_pnl = 0.0
        cross_pnl = 0.0

        if snapshot and snapshot.vix_overlay:
            ov = snapshot.vix_overlay
            # VIX call payoff approximation: delta ~0.3 per contract per VIX point
            vix_pnl = max(0, vix_change) * ov.n_contracts * 100 * 0.3
            # Subtract premium cost (daily amortisation, assume 30 DTE)
            vix_pnl -= ov.estimated_cost / 30.0

        if snapshot and snapshot.delta_hedge:
            dh = snapshot.delta_hedge
            # SPY shares P&L
            delta_pnl = dh.spy_shares * spy_return * (
                snapshot.vix_level / 20.0 if snapshot.vix_level > 0 else 1.0
            )
            # Simplify: shares × SPY price change
            delta_pnl = dh.spy_shares * spy_return

        total_hedge = vix_pnl + delta_pnl + cross_pnl
        total = alpha_pnl + total_hedge
        drag = abs(total_hedge / alpha_pnl) if abs(alpha_pnl) > 1e-9 else 0.0

        return HedgePnL(
            alpha_pnl=alpha_pnl,
            vix_call_pnl=vix_pnl,
            delta_hedge_pnl=delta_pnl,
            cross_hedge_pnl=cross_pnl,
            total_hedge_pnl=total_hedge,
            total_pnl=total,
            hedge_drag_pct=drag,
        )

    def get_history(self) -> HedgeHistory:
        """Return full hedge decision history."""
        cum_pnl = None
        if self._history:
            cum_pnl = HedgePnL()  # zeros
        return HedgeHistory(
            snapshots=list(self._history),
            cumulative_pnl=cum_pnl,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        snapshot: HedgeSnapshot,
        history: Optional[HedgeHistory] = None,
        output_path: str | Path = "reports/dynamic_hedge.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(snapshot, history)
        path.write_text(html, encoding="utf-8")
        logger.info("Dynamic hedge report written to %s", path)
        return path

    # ── VIX call overlay ────────────────────────────────────────────────────
    def _compute_vix_overlay(
        self, portfolio_value: float, vix: float, regime: str,
    ) -> VIXCallOverlay:
        base_pct = self.config.vix_call_budget_pct
        regime_mult = self.config.regime_vix_scale.get(regime, 1.0)
        notional_pct = base_pct * regime_mult

        # VIX call sizing: notional allocated / estimated cost per contract
        # Approximate cost: VIX call ~$2-5 per contract depending on strike
        strike_offset = max(5.0, vix * 0.15)  # ~15% OTM
        est_premium_per_contract = max(100.0, vix * 8.0)  # rough model
        budget = portfolio_value * notional_pct
        n_contracts = max(0, int(budget / est_premium_per_contract))

        # Expected payoff if VIX doubles (tail event)
        vix_at_spike = vix * 2.0
        intrinsic = max(0, vix_at_spike - (vix + strike_offset))
        payoff = n_contracts * intrinsic * 100  # VIX options multiplier

        return VIXCallOverlay(
            notional_pct=notional_pct,
            n_contracts=n_contracts,
            strike_offset=strike_offset,
            estimated_cost=n_contracts * est_premium_per_contract,
            expected_payoff_at_spike=payoff,
            regime_multiplier=regime_mult,
        )

    # ── Delta hedge ─────────────────────────────────────────────────────────
    def _compute_delta_hedge(
        self, portfolio_delta: float, regime: str, spy_price: float,
    ) -> DeltaHedge:
        aggression = self.config.regime_delta_aggression.get(regime, 0.5)
        threshold = self.config.delta_hedge_threshold

        if abs(portfolio_delta) < threshold:
            # Below threshold — no hedge needed
            return DeltaHedge(
                portfolio_delta=portfolio_delta,
                hedge_delta=0.0,
                hedge_ratio=0.0,
                spy_shares=0,
                regime_aggression=aggression,
            )

        # Hedge a fraction of delta based on regime aggression
        hedge_delta = -portfolio_delta * aggression
        spy_shares = int(hedge_delta / spy_price * 10000) if spy_price > 0 else 0
        ratio = aggression if abs(portfolio_delta) >= threshold else 0.0

        return DeltaHedge(
            portfolio_delta=portfolio_delta,
            hedge_delta=hedge_delta,
            hedge_ratio=ratio,
            spy_shares=spy_shares,
            regime_aggression=aggression,
        )

    # ── Cross-hedge ─────────────────────────────────────────────────────────
    def _compute_cross_hedges(
        self,
        returns: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> List[CrossHedge]:
        ids = list(returns.keys())
        if len(ids) < 2:
            return []

        # Build correlation matrix from aligned returns
        frames = {eid: s.rename(eid) for eid, s in returns.items()}
        df = pd.concat(frames.values(), axis=1, join="inner").dropna()
        if len(df) < 20:
            return []

        corr = df.corr()
        hedges: List[CrossHedge] = []

        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                c = float(corr.loc[a, b])
                if abs(c) < self.config.corr_hedge_threshold:
                    continue

                w_a = weights.get(a, 0.0)
                w_b = weights.get(b, 0.0)

                if c > self.config.corr_hedge_threshold:
                    # Highly correlated — reduce both
                    combined_w = w_a + w_b
                    adj = max(0.5, 1.0 - (c - self.config.corr_hedge_threshold))
                    hedges.append(CrossHedge(
                        exp_a=a, exp_b=b, correlation=c,
                        hedge_action="reduce_both",
                        recommended_weight_adj=adj,
                    ))
                elif c < -self.config.corr_hedge_threshold:
                    # Strongly anti-correlated — natural hedge, can increase
                    hedges.append(CrossHedge(
                        exp_a=a, exp_b=b, correlation=c,
                        hedge_action="diversify",
                        recommended_weight_adj=min(1.2, 1.0 + abs(c) * 0.2),
                    ))

        return hedges

    # ── Cost-benefit analysis ───────────────────────────────────────────────
    @staticmethod
    def _compute_cost_benefit(
        portfolio_value: float,
        vix_overlay: VIXCallOverlay,
        delta_hedge: DeltaHedge,
        vix: float,
    ) -> CostBenefit:
        # Annualised hedge cost
        vix_cost_annual = vix_overlay.estimated_cost * 12  # monthly rolls
        delta_cost_annual = abs(delta_hedge.spy_shares) * 0.005 * 252  # ~$0.005/share/day slippage
        total_cost = vix_cost_annual + delta_cost_annual
        cost_pct = total_cost / portfolio_value if portfolio_value > 0 else 0.0

        # Expected DD reduction: VIX overlay provides convex protection
        # Rough model: each $1 of VIX call premium reduces expected DD by ~$3 in a crisis
        vix_dd_reduction = vix_overlay.expected_payoff_at_spike / portfolio_value if portfolio_value > 0 else 0.0
        delta_dd_reduction = abs(delta_hedge.hedge_ratio) * 0.05  # ~5% DD reduction per 100% hedge
        total_dd_reduction = vix_dd_reduction + delta_dd_reduction

        cost_per_point = cost_pct / total_dd_reduction if total_dd_reduction > 1e-9 else float("inf")
        break_even = cost_pct  # hedge pays for itself at this DD level
        net = total_dd_reduction - cost_pct

        if net > 0.02:
            rec = "hedge"
        elif net > 0:
            rec = "partial_hedge"
        else:
            rec = "no_hedge"

        return CostBenefit(
            total_hedge_cost=cost_pct,
            expected_dd_reduction_pct=total_dd_reduction,
            cost_per_dd_point=cost_per_point,
            break_even_dd=break_even,
            net_benefit=net,
            recommendation=rec,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(
        self, snap: HedgeSnapshot, history: Optional[HedgeHistory],
    ) -> str:
        cards = self._html_cards(snap)
        vix_section = self._html_vix_overlay(snap.vix_overlay)
        delta_section = self._html_delta(snap.delta_hedge)
        cross_section = self._html_cross_hedges(snap.cross_hedges)
        cb_section = self._html_cost_benefit(snap.cost_benefit)
        pnl_section = self._html_pnl(snap.hedge_pnl)
        hist_section = self._html_history(history)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Dynamic Hedge Dashboard</title>
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
.rec-badge{{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:700;font-size:.8rem}}
.rec-badge.hedge{{background:#065f46;color:#6ee7b7}}
.rec-badge.partial_hedge{{background:#713f12;color:#fde68a}}
.rec-badge.no_hedge{{background:#7f1d1d;color:#fca5a5}}
</style>
</head>
<body>
<h1>Dynamic Hedge Dashboard</h1>
<p class="sub">Generated {snap.generated_at or 'N/A'} &middot; Regime: <strong>{snap.regime.upper()}</strong> &middot; VIX: <strong>{snap.vix_level:.1f}</strong></p>

{cards}
{vix_section}
{delta_section}
{cross_section}
{cb_section}
{pnl_section}
{hist_section}

</body>
</html>"""

    @staticmethod
    def _html_cards(s: HedgeSnapshot) -> str:
        cb = s.cost_benefit
        rec = cb.recommendation if cb else "N/A"
        cost = f"{cb.total_hedge_cost:.2%}" if cb else "N/A"
        dd_red = f"{cb.expected_dd_reduction_pct:.2%}" if cb else "N/A"
        net = f"{cb.net_benefit:.2%}" if cb else "N/A"
        n_cross = len(s.cross_hedges)
        return f"""<div class="grid">
<div class="card"><div class="lbl">Recommendation</div><div class="val"><span class="rec-badge {rec}">{rec.replace('_',' ').upper()}</span></div></div>
<div class="card"><div class="lbl">Hedge Cost</div><div class="val">{cost}</div></div>
<div class="card"><div class="lbl">DD Reduction</div><div class="val">{dd_red}</div></div>
<div class="card"><div class="lbl">Net Benefit</div><div class="val">{net}</div></div>
<div class="card"><div class="lbl">VIX Contracts</div><div class="val">{s.vix_overlay.n_contracts if s.vix_overlay else 0}</div></div>
<div class="card"><div class="lbl">SPY Shares</div><div class="val">{s.delta_hedge.spy_shares if s.delta_hedge else 0}</div></div>
<div class="card"><div class="lbl">Cross-Hedges</div><div class="val">{n_cross}</div></div>
</div>"""

    @staticmethod
    def _html_vix_overlay(ov: Optional[VIXCallOverlay]) -> str:
        if not ov:
            return ""
        return f"""<div class="sec">
<h2>VIX Call Overlay</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Budget Allocation</td><td>{ov.notional_pct:.2%}</td></tr>
<tr><td>Contracts</td><td>{ov.n_contracts}</td></tr>
<tr><td>Strike Offset</td><td>{ov.strike_offset:.1f} pts OTM</td></tr>
<tr><td>Estimated Cost</td><td>${ov.estimated_cost:,.0f}</td></tr>
<tr><td>Payoff at VIX Spike</td><td>${ov.expected_payoff_at_spike:,.0f}</td></tr>
<tr><td>Regime Multiplier</td><td>{ov.regime_multiplier:.1f}x</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_delta(dh: Optional[DeltaHedge]) -> str:
        if not dh:
            return ""
        return f"""<div class="sec">
<h2>SPY Delta Hedge</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Portfolio Delta</td><td>{dh.portfolio_delta:.4f}</td></tr>
<tr><td>Hedge Delta</td><td>{dh.hedge_delta:.4f}</td></tr>
<tr><td>Hedge Ratio</td><td>{dh.hedge_ratio:.1%}</td></tr>
<tr><td>SPY Shares</td><td>{dh.spy_shares:,}</td></tr>
<tr><td>Regime Aggression</td><td>{dh.regime_aggression:.0%}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_cross_hedges(hedges: List[CrossHedge]) -> str:
        if not hedges:
            return ""
        rows = ""
        for h in hedges:
            cls = "neg" if h.hedge_action == "reduce_both" else "pos"
            rows += (
                f"<tr><td>{h.exp_a}</td><td>{h.exp_b}</td>"
                f"<td>{h.correlation:.3f}</td>"
                f'<td class="{cls}">{h.hedge_action}</td>'
                f"<td>{h.recommended_weight_adj:.2f}x</td></tr>"
            )
        return f"""<div class="sec">
<h2>Correlation Cross-Hedges</h2>
<table>
<thead><tr><th>Exp A</th><th>Exp B</th><th>Correlation</th><th>Action</th><th>Weight Adj</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_cost_benefit(cb: Optional[CostBenefit]) -> str:
        if not cb:
            return ""
        net_cls = "pos" if cb.net_benefit > 0 else "neg"
        return f"""<div class="sec">
<h2>Cost–Benefit Analysis</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Annual Hedge Cost</td><td>{cb.total_hedge_cost:.2%}</td></tr>
<tr><td>Expected DD Reduction</td><td>{cb.expected_dd_reduction_pct:.2%}</td></tr>
<tr><td>Cost per DD Point</td><td>{cb.cost_per_dd_point:.2f}</td></tr>
<tr><td>Break-Even DD</td><td>{cb.break_even_dd:.2%}</td></tr>
<tr><td>Net Benefit</td><td class="{net_cls}">{cb.net_benefit:.2%}</td></tr>
<tr><td>Recommendation</td><td><span class="rec-badge {cb.recommendation}">{cb.recommendation.replace('_',' ').upper()}</span></td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_pnl(pnl: Optional[HedgePnL]) -> str:
        if not pnl:
            return ""
        return f"""<div class="sec">
<h2>Hedge P&L Attribution</h2>
<table>
<thead><tr><th>Component</th><th>P&L</th></tr></thead>
<tbody>
<tr><td>Alpha P&L</td><td class="{'pos' if pnl.alpha_pnl>=0 else 'neg'}">${pnl.alpha_pnl:,.2f}</td></tr>
<tr><td>VIX Call P&L</td><td class="{'pos' if pnl.vix_call_pnl>=0 else 'neg'}">${pnl.vix_call_pnl:,.2f}</td></tr>
<tr><td>Delta Hedge P&L</td><td class="{'pos' if pnl.delta_hedge_pnl>=0 else 'neg'}">${pnl.delta_hedge_pnl:,.2f}</td></tr>
<tr><td>Cross-Hedge P&L</td><td>${pnl.cross_hedge_pnl:,.2f}</td></tr>
<tr><td><strong>Total Hedge P&L</strong></td><td class="{'pos' if pnl.total_hedge_pnl>=0 else 'neg'}"><strong>${pnl.total_hedge_pnl:,.2f}</strong></td></tr>
<tr><td><strong>Total P&L</strong></td><td class="{'pos' if pnl.total_pnl>=0 else 'neg'}"><strong>${pnl.total_pnl:,.2f}</strong></td></tr>
<tr><td>Hedge Drag</td><td>{pnl.hedge_drag_pct:.1%}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_history(hist: Optional[HedgeHistory]) -> str:
        if not hist or not hist.snapshots:
            return ""
        rows = ""
        for s in hist.snapshots[-20:]:  # last 20
            cb = s.cost_benefit
            rec = cb.recommendation if cb else "N/A"
            rows += (
                f"<tr><td>{s.generated_at}</td>"
                f"<td>{s.regime}</td>"
                f"<td>{s.vix_level:.1f}</td>"
                f"<td>{s.vix_overlay.n_contracts if s.vix_overlay else 0}</td>"
                f"<td>{s.delta_hedge.spy_shares if s.delta_hedge else 0}</td>"
                f"<td>{len(s.cross_hedges)}</td>"
                f'<td><span class="rec-badge {rec}">{rec.replace("_"," ")}</span></td></tr>'
            )
        return f"""<div class="sec">
<h2>Hedge Decision History</h2>
<table>
<thead><tr><th>Time</th><th>Regime</th><th>VIX</th><th>VIX Contracts</th><th>SPY Shares</th><th>Cross-Hedges</th><th>Rec</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
