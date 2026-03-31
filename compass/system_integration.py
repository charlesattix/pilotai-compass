"""
Full system integration engine — wires compass modules into an
end-to-end pipeline and verifies each handoff.

Pipeline stages:
  1. Market data ingestion    (prices, volume, VIX)
  2. Regime classification    (Regime enum)
  3. Feature computation      (returns, vol, momentum)
  4. Signal generation        (model prediction)
  5. Position sizing          (risk-budget-aware)
  6. Risk validation          (orchestrator limits)
  7. Portfolio construction   (optimizer)
  8. Hedge overlay            (drawdown protection)
  9. P&L calculation          (daily return attribution)
 10. Performance attribution  (factor decomposition)
 11. Report generation        (HTML summary)

Each stage is independently testable, supports error propagation
and graceful degradation, and records timing + status.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


@dataclass
class StageResult:
    """Result of one pipeline stage."""
    name: str
    status: StageStatus
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Full pipeline execution result."""
    stages: List[StageResult]
    total_duration_ms: float
    n_success: int
    n_failed: int
    n_degraded: int
    final_pnl: float = 0.0
    final_sharpe: float = 0.0


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """Output of market data ingestion stage."""
    prices: pd.Series
    volume: pd.Series
    vix: pd.Series
    returns: pd.Series


@dataclass
class RegimeState:
    """Output of regime classification stage."""
    regime_series: pd.Series
    current_regime: str


@dataclass
class Features:
    """Output of feature computation stage."""
    returns: pd.Series
    volatility: pd.Series
    momentum: pd.Series
    volume_ma: pd.Series


@dataclass
class SignalOutput:
    """Output of signal generation stage."""
    signal: pd.Series
    confidence: pd.Series


@dataclass
class PositionSize:
    """Output of position sizing stage."""
    target_weights: Dict[str, float]
    position_size: float
    risk_budget_used: float


@dataclass
class RiskCheckResult:
    """Output of risk validation stage."""
    approved: bool
    escalation: str
    limits_breached: List[str]
    adjusted_weights: Dict[str, float]


@dataclass
class PortfolioState:
    """Output of portfolio construction stage."""
    weights: Dict[str, float]
    expected_return: float
    expected_vol: float


@dataclass
class HedgeOverlay:
    """Output of hedge overlay stage."""
    hedge_ratio: float
    protection_level: str
    size_multiplier: float
    daily_cost: float


@dataclass
class PnLResult:
    """Output of P&L calculation stage."""
    daily_returns: pd.Series
    cumulative_return: float
    sharpe: float
    max_drawdown: float


@dataclass
class AttributionResult:
    """Output of performance attribution stage."""
    total_return: float
    market_contribution: float
    selection_contribution: float
    residual: float


# ---------------------------------------------------------------------------
# Core pipeline engine
# ---------------------------------------------------------------------------

class SystemIntegration:
    """End-to-end system integration engine.

    Args:
        graceful_degradation: If True, skip failed stages instead of aborting.
    """

    def __init__(self, graceful_degradation: bool = True) -> None:
        self.graceful_degradation = graceful_degradation
        self._stages: List[StageResult] = []

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        name: str,
        fn: Callable,
        dependencies: Optional[List[str]] = None,
        **kwargs,
    ) -> StageResult:
        """Execute a pipeline stage with timing and error handling."""
        deps = dependencies or []

        # Check dependencies
        if deps:
            for dep in deps:
                dep_result = self._get_stage(dep)
                if dep_result and dep_result.status == StageStatus.FAILED:
                    if not self.graceful_degradation:
                        result = StageResult(
                            name=name, status=StageStatus.FAILED,
                            error=f"Dependency '{dep}' failed",
                            dependencies=deps,
                        )
                        self._stages.append(result)
                        return result
                    else:
                        result = StageResult(
                            name=name, status=StageStatus.SKIPPED,
                            error=f"Skipped: dependency '{dep}' failed",
                            dependencies=deps,
                        )
                        self._stages.append(result)
                        return result

        t0 = time.perf_counter()
        try:
            output = fn(**kwargs)
            duration = (time.perf_counter() - t0) * 1000
            result = StageResult(
                name=name, status=StageStatus.SUCCESS,
                duration_ms=duration, output=output, dependencies=deps,
            )
        except Exception as e:
            duration = (time.perf_counter() - t0) * 1000
            if self.graceful_degradation:
                result = StageResult(
                    name=name, status=StageStatus.DEGRADED,
                    duration_ms=duration, error=str(e), dependencies=deps,
                )
            else:
                result = StageResult(
                    name=name, status=StageStatus.FAILED,
                    duration_ms=duration, error=str(e), dependencies=deps,
                )
            logger.warning("Stage '%s' failed: %s", name, e)

        self._stages.append(result)
        return result

    def _get_stage(self, name: str) -> Optional[StageResult]:
        for s in self._stages:
            if s.name == name:
                return s
        return None

    def _get_output(self, name: str) -> Any:
        s = self._get_stage(name)
        return s.output if s and s.status == StageStatus.SUCCESS else None

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    @staticmethod
    def stage_market_data(
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ) -> MarketData:
        """Stage 1: Ingest and validate market data."""
        if prices.empty:
            raise ValueError("Empty price series")
        returns = prices.pct_change().dropna()
        if volume is None:
            volume = pd.Series(1e6, index=prices.index)
        if vix is None:
            vix = pd.Series(20.0, index=prices.index)
        return MarketData(prices=prices, volume=volume, vix=vix, returns=returns)

    @staticmethod
    def stage_regime(vix: pd.Series) -> RegimeState:
        """Stage 2: Classify market regime from VIX."""
        from compass.regime import Regime
        regimes = []
        for v in vix:
            if v > 40:
                regimes.append(Regime.CRASH)
            elif v > 30:
                regimes.append(Regime.HIGH_VOL)
            elif v > 25:
                regimes.append(Regime.BEAR)
            elif v < 15:
                regimes.append(Regime.LOW_VOL)
            else:
                regimes.append(Regime.BULL)
        series = pd.Series(regimes, index=vix.index)
        return RegimeState(regime_series=series, current_regime=str(series.iloc[-1].value))

    @staticmethod
    def stage_features(market: MarketData) -> Features:
        """Stage 3: Compute features from market data."""
        returns = market.returns
        volatility = returns.rolling(21).std() * np.sqrt(TRADING_DAYS)
        momentum = market.prices.pct_change(20)
        volume_ma = market.volume.rolling(20).mean()
        return Features(
            returns=returns, volatility=volatility.fillna(0),
            momentum=momentum.fillna(0), volume_ma=volume_ma.fillna(0),
        )

    @staticmethod
    def stage_signal(features: Features) -> SignalOutput:
        """Stage 4: Generate trading signal from features."""
        # Simple momentum signal
        sig = features.momentum.apply(
            lambda x: 1.0 if x > 0.01 else (-1.0 if x < -0.01 else 0.0))
        conf = features.momentum.abs().clip(0, 0.10) / 0.10
        return SignalOutput(signal=sig, confidence=conf)

    @staticmethod
    def stage_position_size(
        signal: SignalOutput,
        regime: str,
        account_size: float = 100000,
    ) -> PositionSize:
        """Stage 5: Determine position size based on signal and regime."""
        regime_mult = {"bull": 1.0, "low_vol": 1.0, "bear": 0.7,
                        "high_vol": 0.5, "crash": 0.2}.get(regime, 1.0)
        base_size = 0.02 * account_size  # 2% risk
        size = base_size * regime_mult
        latest_sig = float(signal.signal.iloc[-1]) if not signal.signal.empty else 0
        weights = {"SPY": abs(latest_sig) * regime_mult}
        return PositionSize(
            target_weights=weights, position_size=size,
            risk_budget_used=0.02 * regime_mult,
        )

    @staticmethod
    def stage_risk_check(
        position: PositionSize,
        returns: pd.Series,
        equity: float = 100000,
    ) -> RiskCheckResult:
        """Stage 6: Validate position against risk limits."""
        from compass.risk_orchestrator import RiskOrchestrator
        ro = RiskOrchestrator()
        snap = ro.compute_snapshot(returns, equity)
        limits = ro.check_limits(snap)
        breached = [l.name for l in limits if l.breached]
        approved = len(breached) == 0
        adjusted = dict(position.target_weights)
        if not approved:
            adjusted = {k: v * 0.5 for k, v in adjusted.items()}
        return RiskCheckResult(
            approved=approved, escalation=snap.escalation.value,
            limits_breached=breached, adjusted_weights=adjusted,
        )

    @staticmethod
    def stage_portfolio(
        weights: Dict[str, float],
        returns: pd.DataFrame,
    ) -> PortfolioState:
        """Stage 7: Construct portfolio."""
        from compass.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor()
        if returns.shape[1] >= 2:
            pw = pc.risk_parity(returns)
        else:
            pw = pc.equal_weight(returns)
        return PortfolioState(
            weights=pw.weights, expected_return=pw.expected_return,
            expected_vol=pw.expected_vol,
        )

    @staticmethod
    def stage_hedge(
        drawdown: float,
        regime: str,
    ) -> HedgeOverlay:
        """Stage 8: Apply hedge overlay based on drawdown and regime."""
        from compass.drawdown_protection import DrawdownProtection
        dp = DrawdownProtection()
        dp.update(100000)
        state = dp.update(100000 * (1 - drawdown))
        return HedgeOverlay(
            hedge_ratio=state.drawdown * 2,  # proportional hedge
            protection_level=state.level.value,
            size_multiplier=state.size_multiplier,
            daily_cost=state.drawdown * 0.001,
        )

    @staticmethod
    def stage_pnl(
        returns: pd.Series,
        signal: pd.Series,
        hedge_cost: float = 0.0,
    ) -> PnLResult:
        """Stage 9: Calculate P&L from signal × returns."""
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        if aligned.empty:
            return PnLResult(pd.Series(dtype=float), 0, 0, 0)
        strat_ret = aligned["sig"].shift(1).fillna(0) * aligned["ret"] - hedge_cost / TRADING_DAYS
        cum = float((1 + strat_ret).prod() - 1)
        mu = float(strat_ret.mean())
        std = float(strat_ret.std())
        sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
        eq = (1 + strat_ret).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())
        return PnLResult(strat_ret, cum, sharpe, dd)

    @staticmethod
    def stage_attribution(
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
    ) -> AttributionResult:
        """Stage 10: Decompose performance."""
        from compass.performance_attribution import PerformanceAttribution
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(portfolio_returns, market_returns)
        return AttributionResult(
            total_return=fd.total_return,
            market_contribution=fd.market,
            selection_contribution=fd.selection,
            residual=fd.residual,
        )

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
        account_size: float = 100000,
    ) -> PipelineResult:
        """Execute the full end-to-end pipeline."""
        self._stages.clear()
        t0 = time.perf_counter()

        # Stage 1: Market data
        s1 = self._run_stage("market_data", self.stage_market_data,
                              prices=prices, volume=volume, vix=vix)
        market: Optional[MarketData] = s1.output

        # Stage 2: Regime
        s2 = self._run_stage(
            "regime", self.stage_regime,
            dependencies=["market_data"],
            vix=market.vix if market else pd.Series(dtype=float),
        )
        regime_state: Optional[RegimeState] = s2.output

        # Stage 3: Features
        s3 = self._run_stage(
            "features", self.stage_features,
            dependencies=["market_data"],
            market=market if market else MarketData(
                pd.Series(dtype=float), pd.Series(dtype=float),
                pd.Series(dtype=float), pd.Series(dtype=float)),
        )
        features: Optional[Features] = s3.output

        # Stage 4: Signal
        s4 = self._run_stage(
            "signal", self.stage_signal,
            dependencies=["features"],
            features=features if features else Features(
                pd.Series(dtype=float), pd.Series(dtype=float),
                pd.Series(dtype=float), pd.Series(dtype=float)),
        )
        signal_out: Optional[SignalOutput] = s4.output

        # Stage 5: Position sizing
        current_regime = regime_state.current_regime if regime_state else "bull"
        s5 = self._run_stage(
            "position_size", self.stage_position_size,
            dependencies=["signal", "regime"],
            signal=signal_out if signal_out else SignalOutput(
                pd.Series(dtype=float), pd.Series(dtype=float)),
            regime=current_regime,
            account_size=account_size,
        )
        position: Optional[PositionSize] = s5.output

        # Stage 6: Risk check
        rets = market.returns if market else pd.Series(dtype=float)
        s6 = self._run_stage(
            "risk_check", self.stage_risk_check,
            dependencies=["position_size", "market_data"],
            position=position if position else PositionSize({}, 0, 0),
            returns=rets,
            equity=account_size,
        )
        risk_result: Optional[RiskCheckResult] = s6.output

        # Stage 7: Portfolio construction
        ret_df = pd.DataFrame({"SPY": rets}) if not rets.empty else pd.DataFrame()
        s7 = self._run_stage(
            "portfolio", self.stage_portfolio,
            dependencies=["risk_check"],
            weights=risk_result.adjusted_weights if risk_result else {},
            returns=ret_df,
        )

        # Stage 8: Hedge overlay
        dd = 0.0
        if market and not market.prices.empty:
            hwm = market.prices.expanding().max()
            dd = float((1 - market.prices / hwm).iloc[-1])
        s8 = self._run_stage(
            "hedge", self.stage_hedge,
            dependencies=["regime"],
            drawdown=max(dd, 0), regime=current_regime,
        )
        hedge: Optional[HedgeOverlay] = s8.output

        # Stage 9: P&L
        sig_series = signal_out.signal if signal_out else pd.Series(dtype=float)
        hedge_cost = hedge.daily_cost if hedge else 0.0
        s9 = self._run_stage(
            "pnl", self.stage_pnl,
            dependencies=["signal", "market_data", "hedge"],
            returns=rets, signal=sig_series, hedge_cost=hedge_cost,
        )
        pnl: Optional[PnLResult] = s9.output

        # Stage 10: Attribution
        port_rets = pnl.daily_returns if pnl and not pnl.daily_returns.empty else pd.Series(dtype=float)
        s10 = self._run_stage(
            "attribution", self.stage_attribution,
            dependencies=["pnl", "market_data"],
            portfolio_returns=port_rets,
            market_returns=rets,
        )

        total_ms = (time.perf_counter() - t0) * 1000
        n_success = sum(1 for s in self._stages if s.status == StageStatus.SUCCESS)
        n_failed = sum(1 for s in self._stages if s.status == StageStatus.FAILED)
        n_degraded = sum(1 for s in self._stages
                          if s.status in (StageStatus.DEGRADED, StageStatus.SKIPPED))

        return PipelineResult(
            stages=list(self._stages),
            total_duration_ms=total_ms,
            n_success=n_success, n_failed=n_failed, n_degraded=n_degraded,
            final_pnl=pnl.cumulative_return if pnl else 0.0,
            final_sharpe=pnl.sharpe if pnl else 0.0,
        )

    # ------------------------------------------------------------------
    # Run individual stage for testing
    # ------------------------------------------------------------------

    def run_stage_isolated(
        self, name: str, fn: Callable, **kwargs,
    ) -> StageResult:
        """Run a single stage in isolation (for unit testing)."""
        self._stages.clear()
        return self._run_stage(name, fn, **kwargs)

    @property
    def stages(self) -> List[StageResult]:
        return list(self._stages)

    # ------------------------------------------------------------------
    # Error injection for testing
    # ------------------------------------------------------------------

    @staticmethod
    def stage_that_fails(**kwargs) -> None:
        """A stage that always raises for testing error propagation."""
        raise RuntimeError("Intentional test failure")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: PipelineResult,
        output_path: str = "reports/system_integration.html",
    ) -> str:
        """HTML report: data flow, per-stage timing, success/failure."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        status_colors = {
            "success": "#27ae60", "failed": "#e74c3c",
            "degraded": "#e67e22", "skipped": "#999",
            "pending": "#bbb", "running": "#2980b9",
        }

        # Stage table
        stage_rows = []
        for s in result.stages:
            c = status_colors.get(s.status.value, "#999")
            err = f"<br><small>{s.error}</small>" if s.error else ""
            deps = ", ".join(s.dependencies) if s.dependencies else "-"
            stage_rows.append(
                f"<tr><td style='text-align:left'>{s.name}</td>"
                f"<td style='color:{c};font-weight:bold'>{s.status.value.upper()}</td>"
                f"<td>{s.duration_ms:.1f}</td>"
                f"<td style='text-align:left'>{deps}</td>"
                f"<td style='text-align:left'>{err}</td></tr>")

        # Flow diagram (simple SVG)
        stage_names = [s.name for s in result.stages]
        n = len(stage_names)
        flow_w = 720
        box_h, gap = 30, 8
        flow_h = n * (box_h + gap) + 20
        flow_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{flow_w}" '
            f'height="{flow_h}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">']
        for i, s in enumerate(result.stages):
            y = 10 + i * (box_h + gap)
            c = status_colors.get(s.status.value, "#999")
            bw = max(s.duration_ms / max(result.total_duration_ms, 1) * 400, 20)
            flow_parts.append(
                f'<rect x="180" y="{y}" width="{bw:.0f}" height="{box_h}" '
                f'fill="{c}" rx="4"/>')
            flow_parts.append(
                f'<text x="175" y="{y + box_h * 0.7:.0f}" text-anchor="end" '
                f'font-size="11" fill="#333">{s.name}</text>')
            flow_parts.append(
                f'<text x="{185 + bw:.0f}" y="{y + box_h * 0.7:.0f}" '
                f'font-size="10" fill="#666">{s.duration_ms:.1f}ms</text>')
            # Arrow to next
            if i < n - 1:
                flow_parts.append(
                    f'<line x1="380" y1="{y + box_h}" x2="380" '
                    f'y2="{y + box_h + gap}" stroke="#ccc" stroke-width="1" '
                    f'marker-end="url(#arrow)"/>')
        flow_parts.append(
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" '
            'markerWidth="4" markerHeight="4" orient="auto-start-auto">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ccc"/></marker></defs>')
        flow_parts.append("</svg>")
        flow_svg = "\n".join(flow_parts)

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>System Integration</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.big {{ font-size: 1.5em; font-weight: bold; }}
</style></head><body>
<h1>System Integration Report</h1>
<div class="summary">
<p class="big" style="color:{'#27ae60' if result.n_failed == 0 else '#e74c3c'}">
   {result.n_success}/{len(result.stages)} stages passed</p>
<p>Total: {result.total_duration_ms:.0f}ms | Failed: {result.n_failed} |
   Degraded: {result.n_degraded} |
   P&amp;L: {result.final_pnl:+.2%} | Sharpe: {result.final_sharpe:.2f}</p>
</div>

<h2>Pipeline Flow</h2>
{flow_svg}

<h2>Stage Details</h2>
<table><tr><th style='text-align:left'>Stage</th><th>Status</th>
<th>Duration (ms)</th><th style='text-align:left'>Dependencies</th>
<th style='text-align:left'>Error</th></tr>
{''.join(stage_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Integration report -> %s", path)
        return str(path)
