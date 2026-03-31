"""
Unified risk management orchestrator.

Combines portfolio-level risk metrics (VaR, CVaR, Greeks), automated
limit enforcement with escalation, regime-aware adjustment, drawdown
circuit breakers, hedge recommendations, and stress test integration
into a single risk engine.

All methods work on pre-loaded data — no broker connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class EscalationLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    REDUCE = "reduce"
    LIQUIDATE = "liquidate"


@dataclass
class RiskSnapshot:
    date: datetime
    portfolio_var_95: float = 0.0
    portfolio_cvar_95: float = 0.0
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    drawdown: float = 0.0
    escalation: EscalationLevel = EscalationLevel.NORMAL


@dataclass
class RiskLimit:
    name: str
    current: float
    limit: float
    utilisation: float
    breached: bool


@dataclass
class HedgeRecommendation:
    instrument: str
    action: str          # "buy" | "sell"
    quantity: float
    reason: str
    urgency: str         # "low" | "medium" | "high"


@dataclass
class StressResult:
    scenario: str
    pnl_impact: float
    var_impact: float
    worst_asset: str


@dataclass
class DrawdownAlert:
    level: str
    drawdown: float
    action: str
    size_multiplier: float


@dataclass
class OrchestratorReport:
    snapshot: RiskSnapshot
    limits: List[RiskLimit]
    hedges: List[HedgeRecommendation]
    stress_results: List[StressResult]
    drawdown_alert: Optional[DrawdownAlert]
    escalation: EscalationLevel


class RiskOrchestrator:
    """Unified risk management engine.

    Args:
        var_limit: Max portfolio VaR (fraction of NAV).
        drawdown_limits: {level: threshold} for graduated response.
        margin_warning: Margin utilisation warning threshold.
        delta_limit: Max absolute portfolio delta.
    """

    def __init__(
        self,
        var_limit: float = 0.05,
        drawdown_limits: Optional[Dict[str, float]] = None,
        margin_warning: float = 0.80,
        delta_limit: float = 50.0,
    ) -> None:
        self.var_limit = var_limit
        self.drawdown_limits = drawdown_limits or {
            "warning": 0.03, "reduce": 0.05, "liquidate": 0.08,
        }
        self.margin_warning = margin_warning
        self.delta_limit = delta_limit
        self._hwm: float = 0.0
        self._history: List[RiskSnapshot] = []

    # ------------------------------------------------------------------
    # Portfolio risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))

    @staticmethod
    def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        cutoff = np.percentile(returns.dropna(), (1 - confidence) * 100)
        tail = returns[returns <= cutoff]
        return float(-tail.mean()) if not tail.empty else 0.0

    def compute_snapshot(
        self,
        returns: pd.Series,
        equity: float,
        greeks: Optional[Dict[str, float]] = None,
        margin_used: float = 0.0,
        margin_available: float = 0.0,
        date: Optional[datetime] = None,
    ) -> RiskSnapshot:
        if equity > self._hwm:
            self._hwm = equity
        dd = 1 - equity / self._hwm if self._hwm > 0 else 0.0
        g = greeks or {}

        snap = RiskSnapshot(
            date=date or datetime.now(),
            portfolio_var_95=self.compute_var(returns),
            portfolio_cvar_95=self.compute_cvar(returns),
            total_delta=g.get("delta", 0.0),
            total_gamma=g.get("gamma", 0.0),
            total_vega=g.get("vega", 0.0),
            total_theta=g.get("theta", 0.0),
            margin_used=margin_used,
            margin_available=margin_available,
            drawdown=dd,
        )
        snap.escalation = self._determine_escalation(snap)
        self._history.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Limit enforcement
    # ------------------------------------------------------------------

    def check_limits(self, snapshot: RiskSnapshot) -> List[RiskLimit]:
        limits: List[RiskLimit] = []
        # VaR
        var_util = snapshot.portfolio_var_95 / self.var_limit if self.var_limit > 0 else 0
        limits.append(RiskLimit("VaR", snapshot.portfolio_var_95, self.var_limit,
                                 var_util, snapshot.portfolio_var_95 > self.var_limit))
        # Delta
        d_util = abs(snapshot.total_delta) / self.delta_limit if self.delta_limit > 0 else 0
        limits.append(RiskLimit("Delta", abs(snapshot.total_delta), self.delta_limit,
                                 d_util, abs(snapshot.total_delta) > self.delta_limit))
        # Margin
        total_margin = snapshot.margin_used + snapshot.margin_available
        m_util = snapshot.margin_used / total_margin if total_margin > 0 else 0
        limits.append(RiskLimit("Margin", m_util, self.margin_warning,
                                 m_util / self.margin_warning if self.margin_warning > 0 else 0,
                                 m_util > self.margin_warning))
        # Drawdown
        dd_limit = self.drawdown_limits.get("reduce", 0.05)
        limits.append(RiskLimit("Drawdown", snapshot.drawdown, dd_limit,
                                 snapshot.drawdown / dd_limit if dd_limit > 0 else 0,
                                 snapshot.drawdown > dd_limit))
        return limits

    def _determine_escalation(self, snapshot: RiskSnapshot) -> EscalationLevel:
        dd = snapshot.drawdown
        if dd >= self.drawdown_limits.get("liquidate", 0.08):
            return EscalationLevel.LIQUIDATE
        if dd >= self.drawdown_limits.get("reduce", 0.05):
            return EscalationLevel.REDUCE
        if dd >= self.drawdown_limits.get("warning", 0.03):
            return EscalationLevel.WARNING
        if snapshot.portfolio_var_95 > self.var_limit:
            return EscalationLevel.WARNING
        return EscalationLevel.NORMAL

    # ------------------------------------------------------------------
    # Regime adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def regime_risk_multiplier(regime: str) -> float:
        return {"bull": 1.0, "low_vol": 1.1, "bear": 0.7,
                "high_vol": 0.6, "crash": 0.4}.get(regime, 1.0)

    # ------------------------------------------------------------------
    # Drawdown circuit breaker
    # ------------------------------------------------------------------

    def drawdown_alert(self, drawdown: float) -> DrawdownAlert:
        if drawdown >= self.drawdown_limits.get("liquidate", 0.08):
            return DrawdownAlert("RED", drawdown, "liquidate", 0.0)
        if drawdown >= self.drawdown_limits.get("reduce", 0.05):
            return DrawdownAlert("ORANGE", drawdown, "reduce_50pct", 0.5)
        if drawdown >= self.drawdown_limits.get("warning", 0.03):
            return DrawdownAlert("YELLOW", drawdown, "reduce_25pct", 0.75)
        return DrawdownAlert("GREEN", drawdown, "normal", 1.0)

    # ------------------------------------------------------------------
    # Hedge recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def hedge_recommendations(
        snapshot: RiskSnapshot,
        available_instruments: Optional[List[str]] = None,
    ) -> List[HedgeRecommendation]:
        instruments = available_instruments or ["SPY_put", "VIX_call", "TLT"]
        recs: List[HedgeRecommendation] = []
        if abs(snapshot.total_delta) > 30:
            inst = "SPY_put" if snapshot.total_delta > 0 else "SPY_call"
            recs.append(HedgeRecommendation(
                inst, "buy", abs(snapshot.total_delta) * 0.5,
                f"Delta exposure {snapshot.total_delta:.0f}", "high"))
        if snapshot.total_vega > 50:
            recs.append(HedgeRecommendation(
                "VIX_call", "sell", snapshot.total_vega * 0.3,
                f"Vega exposure {snapshot.total_vega:.0f}", "medium"))
        if snapshot.drawdown > 0.03:
            recs.append(HedgeRecommendation(
                "SPY_put", "buy", 10, f"Drawdown {snapshot.drawdown:.1%}", "high"))
        return recs

    # ------------------------------------------------------------------
    # Stress test integration
    # ------------------------------------------------------------------

    @staticmethod
    def run_stress_tests(
        returns: pd.DataFrame,
        weights: Dict[str, float],
        scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[StressResult]:
        if scenarios is None:
            scenarios = {
                "market_crash": {a: -0.10 for a in returns.columns},
                "vol_spike": {a: -0.05 for a in returns.columns},
                "rate_shock": {a: -0.03 for a in returns.columns},
            }
        results: List[StressResult] = []
        for name, shocks in scenarios.items():
            pnl = sum(weights.get(a, 0) * shocks.get(a, 0) for a in returns.columns)
            worst = min(shocks, key=lambda a: shocks.get(a, 0) * weights.get(a, 0))
            results.append(StressResult(name, pnl, abs(pnl) * 1.5, worst))
        return results

    # ------------------------------------------------------------------
    # Full orchestration
    # ------------------------------------------------------------------

    def orchestrate(
        self,
        returns: pd.Series,
        equity: float,
        greeks: Optional[Dict[str, float]] = None,
        margin_used: float = 0.0,
        margin_available: float = 0.0,
        portfolio_returns: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
        date: Optional[datetime] = None,
    ) -> OrchestratorReport:
        snap = self.compute_snapshot(
            returns, equity, greeks, margin_used, margin_available, date)
        limits = self.check_limits(snap)
        hedges = self.hedge_recommendations(snap)
        dd_alert = self.drawdown_alert(snap.drawdown)

        stress: List[StressResult] = []
        if portfolio_returns is not None and weights:
            stress = self.run_stress_tests(portfolio_returns, weights)

        return OrchestratorReport(
            snapshot=snap, limits=limits, hedges=hedges,
            stress_results=stress, drawdown_alert=dd_alert,
            escalation=snap.escalation,
        )

    @property
    def history(self) -> List[RiskSnapshot]:
        return list(self._history)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, report: OrchestratorReport,
        output_path: str = "reports/risk_orchestrator.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = report.snapshot
        esc_colors = {"normal": "#27ae60", "warning": "#f1c40f",
                       "reduce": "#e67e22", "liquidate": "#e74c3c"}
        ec = esc_colors.get(report.escalation.value, "#999")

        limit_rows = [
            f"<tr><td>{l.name}</td><td>{l.current:.4f}</td><td>{l.limit:.4f}</td>"
            f"<td>{l.utilisation:.1%}</td>"
            f"<td style='color:{'#e74c3c' if l.breached else '#27ae60'}'>"
            f"{'BREACH' if l.breached else 'OK'}</td></tr>"
            for l in report.limits
        ]
        hedge_rows = [
            f"<tr><td>{h.instrument}</td><td>{h.action}</td><td>{h.quantity:.1f}</td>"
            f"<td>{h.reason}</td><td>{h.urgency}</td></tr>"
            for h in report.hedges
        ]
        stress_rows = [
            f"<tr><td>{sr.scenario}</td><td>{sr.pnl_impact:+.2%}</td>"
            f"<td>{sr.var_impact:.4f}</td><td>{sr.worst_asset}</td></tr>"
            for sr in report.stress_results
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Risk Orchestrator</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.badge {{ padding: 4px 12px; border-radius: 8px; color: #fff; font-weight: bold; }}
</style></head><body>
<h1>Risk Orchestrator Dashboard</h1>
<div class="summary">
<p><strong>Escalation:</strong> <span class="badge" style="background:{ec}">{report.escalation.value.upper()}</span></p>
<p>VaR: {s.portfolio_var_95:.4f} | CVaR: {s.portfolio_cvar_95:.4f} | DD: {s.drawdown:.2%}</p>
<p>Delta: {s.total_delta:.1f} | Gamma: {s.total_gamma:.2f} | Vega: {s.total_vega:.1f} | Theta: {s.total_theta:.2f}</p>
</div>
<h2>Limit Status</h2>
<table><tr><th>Limit</th><th>Current</th><th>Limit</th><th>Util</th><th>Status</th></tr>
{''.join(limit_rows)}</table>
<h2>Hedge Recommendations</h2>
<table><tr><th>Instrument</th><th>Action</th><th>Qty</th><th>Reason</th><th>Urgency</th></tr>
{''.join(hedge_rows)}</table>
{'<h2>Stress Results</h2><table><tr><th>Scenario</th><th>P&L</th><th>VaR Impact</th><th>Worst</th></tr>' + ''.join(stress_rows) + '</table>' if stress_rows else ''}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
