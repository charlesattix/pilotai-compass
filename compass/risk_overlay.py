"""
Unified Risk Management Overlay.

Combines five risk layers into a single wrapper that can be applied
to any portfolio backtest or live trading system:

  1. Dynamic leverage  — regime-aware 0.5x–2.0x scaling (VIX + term structure + rvol)
  2. Tail risk hedging  — SPY puts + VIX calls sized by crisis score
  3. Event gates        — pre-FOMC/CPI/NFP position scaling (0.5x–1.0x)
  4. Position stops     — per-trade stop-loss enforcement
  5. DD circuit breaker — portfolio-level drawdown cutoff → 0.5x until recovery

Usage:
    overlay = RiskOverlay(config)
    result = overlay.apply(portfolio_returns, spy_returns, vix, vix3m)

All components are optional — disable any layer via config flags.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class RiskOverlayConfig:
    """Master config for all risk layers."""

    # ── Layer toggles ──
    enable_dynamic_leverage: bool = True
    enable_tail_hedge: bool = True
    enable_event_gates: bool = True
    enable_position_stops: bool = True
    enable_dd_circuit_breaker: bool = True

    # ── Dynamic leverage ──
    target_leverage: float = 1.8
    min_leverage: float = 0.3
    vix_calm: float = 15.0
    vix_normal: float = 20.0
    vix_elevated: float = 28.0
    vix_crisis: float = 35.0
    ts_contango: float = 0.90
    ts_flat: float = 1.0
    ts_inverted: float = 1.10
    ts_deep_inversion: float = 1.25
    rvol_low: float = 0.10
    rvol_normal: float = 0.16
    rvol_high: float = 0.25
    rvol_extreme: float = 0.40
    leverage_smoothing_halflife: int = 5

    # ── Tail risk hedge ──
    hedge_annual_cost_budget_pct: float = 2.0
    put_buy_vix_threshold: float = 20.0
    put_base_pct: float = 0.60
    put_payoff_multiplier: float = 12.0
    vix_call_base_pct: float = 0.40
    vix_call_payoff_multiplier: float = 20.0
    vix_call_trigger: float = 25.0
    crisis_vix_threshold: float = 28.0
    crisis_dd_threshold: float = 0.05
    crisis_rvol_threshold: float = 0.30
    crisis_momentum_threshold: float = -0.03
    normal_hedge_ratio: float = 0.30
    crisis_hedge_ratio: float = 0.80

    # ── Event gates ──
    fomc_scaling: Dict[int, float] = field(default_factory=lambda: {
        5: 1.00, 4: 0.90, 3: 0.80, 2: 0.70, 1: 0.60, 0: 0.50,
    })
    cpi_scaling: Dict[int, float] = field(default_factory=lambda: {
        2: 1.00, 1: 0.75, 0: 0.65,
    })
    nfp_scaling: Dict[int, float] = field(default_factory=lambda: {
        2: 1.00, 1: 0.80, 0: 0.75,
    })
    post_fomc_buffer: float = 0.70
    post_cpi_buffer: float = 0.80
    post_nfp_buffer: float = 0.80

    # ── Position stops ──
    stop_loss_pct: float = 0.03         # 3% per-position stop
    trailing_stop_pct: float = 0.05     # 5% trailing stop from peak

    # ── DD circuit breaker ──
    dd_breaker_threshold: float = 0.10  # 10% portfolio drawdown triggers
    dd_breaker_leverage: float = 0.50   # Cut to 0.5x when triggered
    dd_recovery_threshold: float = 0.05 # Must recover to <5% DD to resume


# ── Day state ────────────────────────────────────────────────────────────

class RiskRegime(str, Enum):
    CALM = "calm"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass
class DayRiskState:
    """Complete risk state for a single day."""
    date: Any
    raw_return: float
    adjusted_return: float

    # Dynamic leverage
    leverage: float
    vix: float
    vix_ratio: float
    realized_vol: float
    regime: str

    # Tail hedge
    hedge_cost: float
    hedge_payoff: float
    crisis_score: float
    hedge_active: bool

    # Event gate
    event_scaling: float
    active_events: List[str]

    # Position stops
    stop_triggered: bool
    stop_count: int

    # DD circuit breaker
    drawdown: float
    breaker_active: bool
    portfolio_value: float


# ── Result ───────────────────────────────────────────────────────────────

@dataclass
class RiskOverlayResult:
    """Full backtest result with all risk layers applied."""
    # Portfolio metrics
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    sortino: float
    vol_pct: float
    total_return_pct: float
    n_days: int

    # Comparison vs unprotected
    unprotected_cagr_pct: float
    unprotected_max_dd_pct: float
    unprotected_sharpe: float
    dd_reduction_pct: float

    # Layer stats
    avg_leverage: float
    total_hedge_cost_pct: float
    total_hedge_payoff_pct: float
    net_hedge_cost_pct: float
    event_gate_days: int
    avg_event_scaling: float
    stop_triggers: int
    breaker_activations: int
    breaker_days: int

    # Regime breakdown
    regime_days: Dict[str, int]
    regime_returns: Dict[str, float]

    # Time series
    equity_curve: List[float]
    daily_states: List[DayRiskState]

    # Yearly
    yearly: Dict[int, Dict]


# ── Helper functions ─────────────────────────────────────────────────────

def _ramp(value: float, low: float, high: float) -> float:
    """Piecewise linear: 1.0 at low, 0.0 at high, linear between."""
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return (high - value) / (high - low)


def _compute_metrics(rets: np.ndarray) -> Dict:
    """Standard performance metrics from daily returns."""
    n = len(rets)
    if n < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0, "total_return_pct": 0}

    total = float(np.prod(1 + rets) - 1)
    years = max(n / TRADING_DAYS, 0.5)
    cagr = (1 + total) ** (1 / years) - 1

    vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS)) if np.std(rets) > 1e-10 else 0

    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(dd.max())

    calmar = cagr / max_dd if max_dd > 1e-6 else 0

    down = rets[rets < 0]
    down_std = float(np.std(down, ddof=1) * math.sqrt(TRADING_DAYS)) if len(down) > 1 else vol
    sortino = float(np.mean(rets) * TRADING_DAYS / down_std) if down_std > 1e-10 else 0

    return {
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
        "vol_pct": round(vol * 100, 2),
        "total_return_pct": round(total * 100, 2),
    }


def _crisis_score(
    vix: float, vix_ratio: float, drawdown: float,
    realized_vol: float, momentum_10d: float,
    config: RiskOverlayConfig,
) -> float:
    """Composite crisis score (0–1). Higher = more danger."""
    # VIX component (30%)
    vix_score = min(1.0, max(0.0, (vix - 15) / (config.crisis_vix_threshold - 15)))

    # Term structure (20%): inverted = danger
    ts_score = min(1.0, max(0.0, (vix_ratio - 0.95) / (1.15 - 0.95)))

    # Drawdown (25%)
    dd_score = min(1.0, drawdown / config.crisis_dd_threshold)

    # Realized vol (10%)
    rvol_score = min(1.0, max(0.0, (realized_vol - 0.12) / (config.crisis_rvol_threshold - 0.12)))

    # Momentum (15%): negative momentum = danger
    mom_score = min(1.0, max(0.0, (-momentum_10d) / abs(config.crisis_momentum_threshold)))

    return 0.30 * vix_score + 0.20 * ts_score + 0.25 * dd_score + 0.10 * rvol_score + 0.15 * mom_score


# ── Event gate logic ─────────────────────────────────────────────────────

def _get_event_scaling(
    dt: date, config: RiskOverlayConfig,
) -> Tuple[float, List[str]]:
    """Compute event-gate scaling for a given date.

    Returns (scaling_factor, list_of_active_event_names).
    """
    try:
        from compass.events import get_upcoming_events, compute_composite_scaling
        events = get_upcoming_events(as_of=dt, horizon_days=7)
        scaling = compute_composite_scaling(events)
        names = [e.get("description", e.get("event_type", "event")) for e in events if e.get("days_out", 99) <= 1]
        return scaling, names
    except Exception:
        # Fallback: inline FOMC/CPI/NFP calendar check
        return _inline_event_scaling(dt, config)


def _inline_event_scaling(dt: date, config: RiskOverlayConfig) -> Tuple[float, List[str]]:
    """Simplified inline event gate when compass.events is unavailable."""
    # We use a minimal calendar approach
    scaling = 1.0
    active = []

    # Check day-of-week: NFP is first Friday of each month
    if dt.weekday() == 4 and dt.day <= 7:
        scaling = min(scaling, config.nfp_scaling.get(0, 0.75))
        active.append("NFP")

    # CPI is ~12th of each month
    if 11 <= dt.day <= 13:
        scaling = min(scaling, config.cpi_scaling.get(0, 0.65))
        active.append("CPI")

    return scaling, active


# ── Core overlay engine ──────────────────────────────────────────────────

class RiskOverlay:
    """Unified risk management overlay.

    Wraps any portfolio return series and applies all five risk layers.
    """

    def __init__(self, config: Optional[RiskOverlayConfig] = None):
        self.config = config or RiskOverlayConfig()

    def apply(
        self,
        portfolio_returns: pd.Series,
        spy_returns: pd.Series,
        vix: pd.Series,
        vix3m: Optional[pd.Series] = None,
        starting_capital: float = 100_000.0,
    ) -> RiskOverlayResult:
        """Apply all risk layers to a portfolio return series.

        Args:
            portfolio_returns: Daily returns (decimal, e.g. 0.01 = +1%)
            spy_returns: SPY daily returns (for beta/hedge computation)
            vix: Daily VIX close series
            vix3m: Daily VIX3M close series (optional; defaults to VIX * 0.9)
            starting_capital: Initial portfolio value

        Returns:
            RiskOverlayResult with full metrics and daily state.
        """
        cfg = self.config
        n = len(portfolio_returns)

        # Align all series
        common_idx = portfolio_returns.index
        port_rets = portfolio_returns.reindex(common_idx).fillna(0).values
        spy_rets = spy_returns.reindex(common_idx).fillna(0).values
        vix_vals = vix.reindex(common_idx).ffill().fillna(18).values
        if vix3m is not None:
            vix3m_vals = vix3m.reindex(common_idx).ffill().fillna(16).values
        else:
            vix3m_vals = vix_vals * 0.9

        dates = common_idx

        # Pre-compute rolling stats
        rvol_20d = self._rolling_vol(port_rets, 20)
        momentum_10d = self._rolling_sum(port_rets, 10)

        # State tracking
        equity = starting_capital
        peak_equity = starting_capital
        adjusted_returns = np.zeros(n)
        states: List[DayRiskState] = []
        equity_curve = [starting_capital]

        # Smoothed leverage (EMA)
        prev_leverage = cfg.target_leverage

        # Circuit breaker state
        breaker_active = False
        breaker_activations = 0
        breaker_days = 0

        # Stop tracking
        position_peak = 0.0
        stop_triggers = 0

        # Event tracking
        event_gate_days = 0
        event_scaling_sum = 0.0

        # Regime counters
        regime_days: Dict[str, int] = {}
        regime_rets: Dict[str, List[float]] = {}

        for i in range(n):
            raw_ret = port_rets[i]
            vix_val = float(vix_vals[i])
            vix3m_val = float(vix3m_vals[i])
            vix_ratio = vix_val / max(vix3m_val, 1.0)
            rv = float(rvol_20d[i])
            mom = float(momentum_10d[i])
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

            # ── Layer 1: Dynamic leverage ──
            if cfg.enable_dynamic_leverage:
                vix_scale = _ramp(vix_val, cfg.vix_calm, cfg.vix_crisis)
                ts_scale = _ramp(vix_ratio, cfg.ts_contango, cfg.ts_deep_inversion)
                rvol_scale = _ramp(rv, cfg.rvol_low, cfg.rvol_extreme)
                raw_leverage = cfg.target_leverage * vix_scale * ts_scale * rvol_scale
                raw_leverage = max(cfg.min_leverage, min(cfg.target_leverage, raw_leverage))

                # EMA smoothing
                alpha = 1 - 0.5 ** (1 / max(cfg.leverage_smoothing_halflife, 1))
                leverage = alpha * raw_leverage + (1 - alpha) * prev_leverage
                prev_leverage = leverage
            else:
                leverage = 1.0

            # Classify regime
            regime = self._classify_regime(vix_val, rv, dd, cfg)

            # ── Layer 2: Tail risk hedge ──
            hedge_cost = 0.0
            hedge_payoff = 0.0
            c_score = 0.0
            hedge_active = False

            if cfg.enable_tail_hedge:
                c_score = _crisis_score(vix_val, vix_ratio, dd, rv, mom, cfg)

                # Daily hedge budget
                daily_budget = (cfg.hedge_annual_cost_budget_pct / 100) * equity / TRADING_DAYS

                # Scale budget by crisis score
                budget_scale = 1.0 + c_score * 2.0  # up to 3x budget in crisis
                daily_budget *= budget_scale

                # Hedge ratio
                hedge_ratio = cfg.normal_hedge_ratio + c_score * (cfg.crisis_hedge_ratio - cfg.normal_hedge_ratio)

                # SPY put cost and payoff
                put_cost = daily_budget * cfg.put_base_pct
                put_payoff_val = 0.0
                spy_ret = spy_rets[i]
                if spy_ret < -0.005:  # down > 0.5%
                    severity = abs(spy_ret) / 0.03
                    put_payoff_val = put_cost * cfg.put_payoff_multiplier * severity * hedge_ratio

                # VIX call cost and payoff
                vix_cost = daily_budget * cfg.vix_call_base_pct
                vix_payoff_val = 0.0
                if i > 0:
                    prev_vix = float(vix_vals[i - 1])
                    vix_change = (vix_val - prev_vix) / max(prev_vix, 1.0)
                    if vix_change > 0.05:  # VIX spike > 5%
                        vix_payoff_val = vix_cost * cfg.vix_call_payoff_multiplier * vix_change * hedge_ratio

                hedge_cost = (put_cost + vix_cost) / max(equity, 1)
                hedge_payoff = (put_payoff_val + vix_payoff_val) / max(equity, 1)
                hedge_active = hedge_payoff > hedge_cost * 0.5

            # ── Layer 3: Event gates ──
            event_scaling = 1.0
            active_events: List[str] = []

            if cfg.enable_event_gates:
                try:
                    dt = dates[i]
                    if hasattr(dt, 'date'):
                        dt = dt.date()
                    event_scaling, active_events = _get_event_scaling(dt, cfg)
                except Exception:
                    event_scaling = 1.0

                if event_scaling < 1.0:
                    event_gate_days += 1
                event_scaling_sum += event_scaling
            else:
                event_scaling_sum += 1.0

            # ── Layer 4: Position stops ──
            stop_triggered = False

            if cfg.enable_position_stops:
                # Track cumulative trade P&L for stop
                if raw_ret > 0:
                    position_peak = max(position_peak, position_peak + raw_ret)
                else:
                    # Check fixed stop
                    if raw_ret < -cfg.stop_loss_pct:
                        stop_triggered = True
                        stop_triggers += 1

                    # Check trailing stop from position peak
                    if position_peak > 0 and raw_ret < -cfg.trailing_stop_pct:
                        stop_triggered = True
                        stop_triggers += 1

                # Reset position tracking periodically (every ~14 days)
                if i % 14 == 0:
                    position_peak = 0.0

            # ── Layer 5: DD circuit breaker ──
            if cfg.enable_dd_circuit_breaker:
                if not breaker_active and dd >= cfg.dd_breaker_threshold:
                    breaker_active = True
                    breaker_activations += 1
                    logger.info("DD breaker ACTIVATED at %.1f%% DD on %s", dd * 100, dates[i])
                elif breaker_active and dd < cfg.dd_recovery_threshold:
                    breaker_active = False
                    logger.info("DD breaker RELEASED at %.1f%% DD on %s", dd * 100, dates[i])

                if breaker_active:
                    breaker_days += 1

            # ── Combine all layers ──
            effective_leverage = leverage

            # Event gate reduces leverage
            effective_leverage *= event_scaling

            # Circuit breaker override
            if breaker_active:
                effective_leverage = min(effective_leverage, cfg.dd_breaker_leverage)

            # Stop triggered — zero out return for this day
            if stop_triggered:
                adjusted_ret = -cfg.stop_loss_pct * effective_leverage
            else:
                adjusted_ret = raw_ret * effective_leverage

            # Add hedge effect (payoff minus cost)
            adjusted_ret += (hedge_payoff - hedge_cost)

            adjusted_returns[i] = adjusted_ret
            equity *= (1 + adjusted_ret)
            peak_equity = max(peak_equity, equity)
            equity_curve.append(equity)

            # Track regime stats
            regime_days[regime] = regime_days.get(regime, 0) + 1
            if regime not in regime_rets:
                regime_rets[regime] = []
            regime_rets[regime].append(adjusted_ret)

            states.append(DayRiskState(
                date=dates[i],
                raw_return=round(raw_ret, 6),
                adjusted_return=round(adjusted_ret, 6),
                leverage=round(effective_leverage, 4),
                vix=round(vix_val, 2),
                vix_ratio=round(vix_ratio, 4),
                realized_vol=round(rv, 4),
                regime=regime,
                hedge_cost=round(hedge_cost, 6),
                hedge_payoff=round(hedge_payoff, 6),
                crisis_score=round(c_score, 4),
                hedge_active=hedge_active,
                event_scaling=round(event_scaling, 4),
                active_events=active_events,
                stop_triggered=stop_triggered,
                stop_count=stop_triggers,
                drawdown=round(dd, 6),
                breaker_active=breaker_active,
                portfolio_value=round(equity, 2),
            ))

        # ── Compute final metrics ──
        protected = _compute_metrics(adjusted_returns)
        unprotected = _compute_metrics(port_rets)

        # Regime returns
        regime_return_map = {}
        for r, rl in regime_rets.items():
            arr = np.array(rl)
            if len(arr) > 0:
                regime_return_map[r] = round(float(np.sum(arr)) * 100, 2)
            else:
                regime_return_map[r] = 0.0

        # Yearly
        yearly = self._yearly_breakdown(adjusted_returns, dates)

        # Hedge totals
        total_cost = sum(s.hedge_cost for s in states)
        total_payoff = sum(s.hedge_payoff for s in states)

        return RiskOverlayResult(
            cagr_pct=protected["cagr_pct"],
            sharpe=protected["sharpe"],
            max_dd_pct=protected["max_dd_pct"],
            calmar=protected["calmar"],
            sortino=protected["sortino"],
            vol_pct=protected["vol_pct"],
            total_return_pct=protected["total_return_pct"],
            n_days=n,
            unprotected_cagr_pct=unprotected["cagr_pct"],
            unprotected_max_dd_pct=unprotected["max_dd_pct"],
            unprotected_sharpe=unprotected["sharpe"],
            dd_reduction_pct=round(unprotected["max_dd_pct"] - protected["max_dd_pct"], 2),
            avg_leverage=round(float(np.mean([s.leverage for s in states])), 3),
            total_hedge_cost_pct=round(total_cost * 100, 3),
            total_hedge_payoff_pct=round(total_payoff * 100, 3),
            net_hedge_cost_pct=round((total_cost - total_payoff) * 100, 3),
            event_gate_days=event_gate_days,
            avg_event_scaling=round(event_scaling_sum / max(n, 1), 4),
            stop_triggers=stop_triggers,
            breaker_activations=breaker_activations,
            breaker_days=breaker_days,
            regime_days=regime_days,
            regime_returns=regime_return_map,
            equity_curve=equity_curve,
            daily_states=states,
            yearly=yearly,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
        """20-day annualized rolling volatility."""
        n = len(rets)
        out = np.full(n, 0.15)  # default
        for i in range(window, n):
            chunk = rets[i - window:i]
            out[i] = float(np.std(chunk, ddof=1) * math.sqrt(TRADING_DAYS))
        return out

    @staticmethod
    def _rolling_sum(rets: np.ndarray, window: int) -> np.ndarray:
        """Rolling sum of returns (momentum proxy)."""
        n = len(rets)
        out = np.zeros(n)
        for i in range(window, n):
            out[i] = float(np.sum(rets[i - window:i]))
        return out

    @staticmethod
    def _classify_regime(vix: float, rvol: float, dd: float,
                         cfg: RiskOverlayConfig) -> str:
        if vix >= cfg.vix_crisis or dd >= cfg.dd_breaker_threshold:
            return RiskRegime.CRISIS.value
        if vix >= cfg.vix_elevated or rvol >= cfg.rvol_high:
            return RiskRegime.ELEVATED.value
        if vix >= cfg.vix_normal:
            return RiskRegime.NORMAL.value
        return RiskRegime.CALM.value

    @staticmethod
    def _yearly_breakdown(rets: np.ndarray, dates) -> Dict[int, Dict]:
        """Year-by-year performance stats."""
        df = pd.DataFrame({"ret": rets}, index=dates)
        df["year"] = df.index.year
        yearly = {}
        for yr, grp in df.groupby("year"):
            r = grp["ret"].values
            m = _compute_metrics(r)
            yearly[int(yr)] = m
        return yearly


# ── Report generator ─────────────────────────────────────────────────────

def generate_report(result: RiskOverlayResult, output_path: str = "reports/risk_overlay_spec.html") -> str:
    """Generate HTML report documenting the full risk framework."""

    # Regime table
    regime_rows = ""
    for regime in ["calm", "normal", "elevated", "crisis"]:
        days = result.regime_days.get(regime, 0)
        pct = days / max(result.n_days, 1) * 100
        ret = result.regime_returns.get(regime, 0)
        rc = "#3fb950" if ret > 0 else "#ef4444"
        regime_rows += f"<tr><td>{regime}</td><td>{days}</td><td>{pct:.1f}%</td><td style='color:{rc}'>{ret:+.2f}%</td></tr>\n"

    # Yearly table
    yearly_rows = ""
    for yr in sorted(result.yearly.keys()):
        y = result.yearly[yr]
        c = "#3fb950" if y["cagr_pct"] > 0 else "#ef4444"
        yearly_rows += (
            f"<tr><td>{yr}</td><td style='color:{c}'>{y['cagr_pct']:+.1f}%</td>"
            f"<td>{y['sharpe']:.2f}</td><td style='color:#f59e0b'>{y['max_dd_pct']:.1f}%</td>"
            f"<td>{y['vol_pct']:.1f}%</td></tr>\n"
        )

    # DD reduction verdict
    if result.dd_reduction_pct > 5:
        dd_verdict = "STRONG"
        dd_color = "#3fb950"
    elif result.dd_reduction_pct > 0:
        dd_verdict = "POSITIVE"
        dd_color = "#d29922"
    else:
        dd_verdict = "NONE"
        dd_color = "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Unified Risk Overlay Specification</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2,h3{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid #58a6ff;border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:1.6em;font-weight:800;color:#58a6ff}}
.hero .sub{{color:#8b949e;margin-top:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:#8b949e;font-size:.75em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.05em;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.82em}}
th,td{{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22;font-size:.75em;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
tr:hover td{{background:#161b2280}}
.section{{margin:36px 0}}
.note{{color:#8b949e;font-size:.82em;margin:6px 0}}
.layer{{background:#161b22;border-left:4px solid #58a6ff;padding:14px;margin:14px 0;border-radius:4px}}
.layer h4{{margin:0 0 8px 0;color:#58a6ff;font-size:.95em}}
.warn{{border-left-color:#f59e0b}} .win{{border-left-color:#3fb950}} .fail{{border-left-color:#ef4444}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
code{{background:#161b22;padding:2px 6px;border-radius:4px;font-size:.85em;color:#f0883e}}
</style></head><body>

<h1>Unified Risk Management Overlay</h1>
<p class="note">Five-layer risk framework &bull; Dynamic leverage + Tail hedge + Event gates + Stops + DD breaker</p>

<div class="hero">
  <div class="big">DD Reduction: {dd_verdict} ({result.dd_reduction_pct:+.1f}pp)</div>
  <div class="sub">
    Protected: {result.cagr_pct:.1f}% CAGR / {result.max_dd_pct:.1f}% DD / Sharpe {result.sharpe:.2f} |
    Unprotected: {result.unprotected_cagr_pct:.1f}% CAGR / {result.unprotected_max_dd_pct:.1f}% DD / Sharpe {result.unprotected_sharpe:.2f}
  </div>
</div>

<div class="cards">
  <div class="c"><div class="l">Protected CAGR</div><div class="v" style="color:#3fb950">{result.cagr_pct:.1f}%</div></div>
  <div class="c"><div class="l">Protected Max DD</div><div class="v" style="color:#f59e0b">{result.max_dd_pct:.1f}%</div></div>
  <div class="c"><div class="l">Protected Sharpe</div><div class="v">{result.sharpe:.2f}</div></div>
  <div class="c"><div class="l">Avg Leverage</div><div class="v">{result.avg_leverage:.2f}x</div></div>
  <div class="c"><div class="l">Net Hedge Cost</div><div class="v">{result.net_hedge_cost_pct:.1f}%</div></div>
  <div class="c"><div class="l">Event Gate Days</div><div class="v">{result.event_gate_days}</div></div>
  <div class="c"><div class="l">Stop Triggers</div><div class="v">{result.stop_triggers}</div></div>
  <div class="c"><div class="l">Breaker Activations</div><div class="v">{result.breaker_activations} ({result.breaker_days}d)</div></div>
</div>

<!-- Layer Specification -->
<div class="section">
<h2>Risk Layer Specification</h2>

<div class="layer win">
<h4>Layer 1: Dynamic Leverage (Regime-Aware 0.3x&ndash;1.8x)</h4>
<p>Leverage = <code>target &times; vix_scale &times; ts_scale &times; rvol_scale</code>, EMA-smoothed (halflife={result.daily_states[0].leverage if result.daily_states else 'N/A'}d).</p>
<table>
<thead><tr><th>VIX Zone</th><th>Threshold</th><th>Leverage Scale</th></tr></thead>
<tbody>
<tr><td>Calm</td><td>&lt; 15</td><td>1.0x (full)</td></tr>
<tr><td>Normal</td><td>&lt; 20</td><td>~0.7x</td></tr>
<tr><td>Elevated</td><td>&lt; 28</td><td>~0.4x</td></tr>
<tr><td>Crisis</td><td>&ge; 35</td><td>0.0x (min floor)</td></tr>
</tbody></table>
<p>Also scales on VIX/VIX3M term structure (contango=full, deep inversion=min) and 20-day realized vol.</p>
</div>

<div class="layer">
<h4>Layer 2: Tail Risk Hedging (SPY Puts + VIX Calls)</h4>
<p>Annual budget: <strong>2%</strong> of portfolio. Allocates 60% to SPY puts, 40% to VIX calls.
Budget scales up to 3x in crisis (crisis_score=1.0). Hedge ratio: 30% normal, 80% crisis.</p>
<ul>
<li><strong>SPY puts</strong>: 20-delta OTM puts. Payoff triggers on SPY drops &gt; 0.5%. 12x convex multiplier.</li>
<li><strong>VIX calls</strong>: Payoff triggers on VIX spikes &gt; 5%. 20x convex multiplier for tail events.</li>
<li><strong>Crisis score</strong>: Composite of VIX (30%), term structure (20%), DD (25%), realized vol (10%), momentum (15%).</li>
</ul>
</div>

<div class="layer warn">
<h4>Layer 3: Event Gates (FOMC / CPI / NFP)</h4>
<p>Automatic position scaling around macro events:</p>
<table>
<thead><tr><th>Event</th><th>5d</th><th>4d</th><th>3d</th><th>2d</th><th>1d</th><th>Day-of</th><th>+1d</th></tr></thead>
<tbody>
<tr><td>FOMC</td><td>1.0x</td><td>0.9x</td><td>0.8x</td><td>0.7x</td><td>0.6x</td><td>0.5x</td><td>0.7x</td></tr>
<tr><td>CPI</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>1.0x</td><td>0.75x</td><td>0.65x</td><td>0.8x</td></tr>
<tr><td>NFP</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>1.0x</td><td>0.8x</td><td>0.75x</td><td>0.8x</td></tr>
</tbody></table>
<p>Multiple events: per-type minimum applied (not multiplicative).</p>
</div>

<div class="layer">
<h4>Layer 4: Position-Level Stop Losses</h4>
<ul>
<li><strong>Fixed stop</strong>: Exit if single-day loss exceeds 3%</li>
<li><strong>Trailing stop</strong>: Exit if drawdown from position peak exceeds 5%</li>
<li>Position tracking resets every 14 days (rolling trade windows)</li>
</ul>
</div>

<div class="layer fail">
<h4>Layer 5: Portfolio DD Circuit Breaker</h4>
<ul>
<li><strong>Trigger</strong>: Portfolio drawdown hits 10% &rarr; cut leverage to 0.5x</li>
<li><strong>Recovery</strong>: DD must recover below 5% before resuming normal leverage</li>
<li>Prevents catastrophic ruin during extended drawdowns</li>
<li>Hysteresis gap (10% on / 5% off) prevents whipsaw</li>
</ul>
</div>
</div>

<!-- Performance Comparison -->
<div class="section">
<h2>Performance: Protected vs Unprotected</h2>
<div class="grid2">
<div>
<table>
<thead><tr><th>Metric</th><th>Protected</th><th>Unprotected</th><th>Delta</th></tr></thead>
<tbody>
<tr><td>CAGR</td><td>{result.cagr_pct:.1f}%</td><td>{result.unprotected_cagr_pct:.1f}%</td><td>{result.cagr_pct - result.unprotected_cagr_pct:+.1f}pp</td></tr>
<tr><td>Max DD</td><td style='color:#f59e0b'>{result.max_dd_pct:.1f}%</td><td>{result.unprotected_max_dd_pct:.1f}%</td><td style='color:{dd_color}'>{result.dd_reduction_pct:+.1f}pp</td></tr>
<tr><td>Sharpe</td><td>{result.sharpe:.2f}</td><td>{result.unprotected_sharpe:.2f}</td><td>{result.sharpe - result.unprotected_sharpe:+.2f}</td></tr>
<tr><td>Calmar</td><td>{result.calmar:.2f}</td><td>&mdash;</td><td>&mdash;</td></tr>
<tr><td>Sortino</td><td>{result.sortino:.2f}</td><td>&mdash;</td><td>&mdash;</td></tr>
<tr><td>Vol</td><td>{result.vol_pct:.1f}%</td><td>&mdash;</td><td>&mdash;</td></tr>
</tbody></table>
</div>
<div>
<h3>Regime Distribution</h3>
<table>
<thead><tr><th>Regime</th><th>Days</th><th>% of Total</th><th>Contribution</th></tr></thead>
<tbody>{regime_rows}</tbody></table>
</div>
</div>
</div>

<!-- Yearly -->
<div class="section">
<h2>Year-by-Year Performance</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>{yearly_rows}</tbody></table>
</div>

<!-- Layer Impact Summary -->
<div class="section">
<h2>Layer Impact Summary</h2>
<table>
<thead><tr><th>Layer</th><th>Active Days</th><th>Impact</th></tr></thead>
<tbody>
<tr><td>Dynamic Leverage</td><td>{result.n_days}</td><td>Avg leverage {result.avg_leverage:.2f}x (target 1.8x)</td></tr>
<tr><td>Tail Hedge</td><td>{result.n_days}</td><td>Cost {result.total_hedge_cost_pct:.1f}%, Payoff {result.total_hedge_payoff_pct:.1f}%, Net {result.net_hedge_cost_pct:+.1f}%</td></tr>
<tr><td>Event Gates</td><td>{result.event_gate_days}</td><td>Avg scaling {result.avg_event_scaling:.3f}</td></tr>
<tr><td>Position Stops</td><td>{result.stop_triggers}</td><td>{result.stop_triggers} stop triggers</td></tr>
<tr><td>DD Circuit Breaker</td><td>{result.breaker_days}</td><td>{result.breaker_activations} activations, {result.breaker_days} days at 0.5x</td></tr>
</tbody></table>
</div>

<p class="note" style="margin-top:40px;text-align:center">
  Unified Risk Overlay &bull; compass/risk_overlay.py &bull; PilotAI &bull; {datetime.now().strftime('%Y-%m-%d')}
</p>
</body></html>"""

    from pathlib import Path
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)


# ── Convenience: generate synthetic data for testing ─────────────────────

def generate_test_data(
    n_days: int = 1260,  # ~5 years
    base_cagr: float = 0.10,
    base_vol: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.Series]:
    """Generate synthetic market data for backtesting the overlay."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    daily_ret = base_cagr / TRADING_DAYS
    daily_vol = base_vol / math.sqrt(TRADING_DAYS)

    # Portfolio returns with regime shifts
    rets = rng.normal(daily_ret, daily_vol, n_days)

    # Add a crash (days 50-60)
    if n_days > 55:
        rets[50:55] = rng.normal(-0.03, 0.02, 5)
    # Add a moderate drawdown (days 400-420)
    if n_days > 420:
        rets[400:420] = rng.normal(-0.005, 0.01, 20)

    # SPY returns (correlated but not identical)
    spy_rets = 0.7 * rets + 0.3 * rng.normal(daily_ret * 0.8, daily_vol, n_days)

    # VIX (inverse relationship with returns)
    vix_base = 18.0
    vix = np.full(n_days, vix_base)
    for i in range(1, n_days):
        shock = -rets[i] * 200 + rng.normal(0, 0.5)
        vix[i] = max(10, min(80, vix[i-1] * 0.95 + vix_base * 0.05 + shock))

    # VIX3M (slightly lower, smoother)
    vix3m = vix * 0.9 + rng.normal(0, 0.3, n_days)
    vix3m = np.maximum(vix3m, 9)

    return {
        "portfolio_returns": pd.Series(rets, index=dates),
        "spy_returns": pd.Series(spy_rets, index=dates),
        "vix": pd.Series(vix, index=dates),
        "vix3m": pd.Series(vix3m, index=dates),
    }


# ── CLI runner ───────────────────────────────────────────────────────────

def run_demo():
    """Run the overlay on synthetic data and generate report."""
    print("Generating test data...")
    data = generate_test_data()
    print(f"  {len(data['portfolio_returns'])} trading days")

    print("Applying risk overlay...")
    overlay = RiskOverlay()
    result = overlay.apply(
        data["portfolio_returns"], data["spy_returns"],
        data["vix"], data["vix3m"],
    )

    print(f"\n=== Results ===")
    print(f"  Protected:   CAGR {result.cagr_pct:.1f}%, DD {result.max_dd_pct:.1f}%, Sharpe {result.sharpe:.2f}")
    print(f"  Unprotected: CAGR {result.unprotected_cagr_pct:.1f}%, DD {result.unprotected_max_dd_pct:.1f}%, Sharpe {result.unprotected_sharpe:.2f}")
    print(f"  DD reduction: {result.dd_reduction_pct:+.1f}pp")
    print(f"  Avg leverage: {result.avg_leverage:.2f}x")
    print(f"  Hedge net cost: {result.net_hedge_cost_pct:.1f}%")
    print(f"  Event gate days: {result.event_gate_days}")
    print(f"  Stop triggers: {result.stop_triggers}")
    print(f"  Breaker activations: {result.breaker_activations} ({result.breaker_days}d)")
    print(f"  Regimes: {result.regime_days}")

    print("\nGenerating report...")
    path = generate_report(result)
    print(f"Report saved to {path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo()
