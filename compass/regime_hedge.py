"""
Regime-adaptive hedging engine.

Dynamically adjusts hedge parameters based on the detected market regime,
with smooth exponential blending during transitions.

Per-regime hedge parameter profiles:
  BULL     — minimal hedging (low cost, wide stops)
  LOW_VOL  — light hedging (VIX puts, moderate stops)
  BEAR     — moderate hedging (tighter stops, higher HV scale)
  HIGH_VOL — heavy hedging (VIX calls, tight stops)
  CRASH    — maximum hedging (VIX floor activated, emergency stops)

Integrates with compass.regime.Regime for detection.

All methods work on pre-loaded data — no broker connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.regime import Regime

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HedgeParams:
    """Hedge parameter set for a single regime."""
    vix_floor: float = 0.0        # min VIX-equivalent hedge level
    vix_ceiling: float = 1.0      # max VIX-equivalent hedge level
    stop_multiplier: float = 2.0  # ATR multiplier for trailing stop
    hv_scale: float = 1.0         # historical vol scaling factor
    hedge_ratio: float = 0.0      # fraction of portfolio hedged
    cost_budget: float = 0.005    # max annualised cost (fraction of NAV)


DEFAULT_PROFILES: Dict[Regime, HedgeParams] = {
    Regime.BULL: HedgeParams(
        vix_floor=0.0, vix_ceiling=0.20, stop_multiplier=3.0,
        hv_scale=0.8, hedge_ratio=0.05, cost_budget=0.002),
    Regime.LOW_VOL: HedgeParams(
        vix_floor=0.0, vix_ceiling=0.15, stop_multiplier=2.5,
        hv_scale=0.9, hedge_ratio=0.10, cost_budget=0.003),
    Regime.BEAR: HedgeParams(
        vix_floor=0.10, vix_ceiling=0.35, stop_multiplier=1.8,
        hv_scale=1.2, hedge_ratio=0.25, cost_budget=0.008),
    Regime.HIGH_VOL: HedgeParams(
        vix_floor=0.15, vix_ceiling=0.50, stop_multiplier=1.5,
        hv_scale=1.5, hedge_ratio=0.35, cost_budget=0.012),
    Regime.CRASH: HedgeParams(
        vix_floor=0.25, vix_ceiling=1.0, stop_multiplier=1.0,
        hv_scale=2.0, hedge_ratio=0.50, cost_budget=0.020),
}


@dataclass
class HedgeState:
    """Snapshot of hedge parameters at a point in time."""
    date: datetime
    regime: Regime
    params: HedgeParams
    blended: bool = False     # True if currently transitioning
    blend_alpha: float = 1.0  # 1.0 = fully in new regime
    daily_cost: float = 0.0


@dataclass
class RegimeTransition:
    """Recorded regime transition."""
    date: datetime
    from_regime: Regime
    to_regime: Regime
    transition_days: int


@dataclass
class HedgeCostSummary:
    """Hedge cost tracking per regime."""
    regime: str
    n_days: int
    total_cost: float
    avg_daily_cost: float
    max_daily_cost: float
    cost_as_pct_return: float


@dataclass
class BacktestComparison:
    """Static vs adaptive hedging comparison."""
    adaptive_pnl: float
    static_pnl: float
    adaptive_max_dd: float
    static_max_dd: float
    adaptive_sharpe: float
    static_sharpe: float
    adaptive_cost: float
    static_cost: float
    improvement_pnl: float
    improvement_dd: float


@dataclass
class GridSweepResult:
    """Result of parameter grid sweep for one regime."""
    regime: str
    best_hedge_ratio: float
    best_stop_mult: float
    best_sharpe: float
    all_results: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class RegimeHedgeEngine:
    """Regime-adaptive hedging engine.

    Args:
        profiles: Per-regime hedge parameter overrides.
        transition_halflife: Days for exponential blending during transitions.
        static_hedge_ratio: Hedge ratio for static (non-adaptive) baseline.
    """

    def __init__(
        self,
        profiles: Optional[Dict[Regime, HedgeParams]] = None,
        transition_halflife: int = 5,
        static_hedge_ratio: float = 0.15,
    ) -> None:
        self.profiles = profiles or dict(DEFAULT_PROFILES)
        self.transition_halflife = transition_halflife
        self.static_hedge_ratio = static_hedge_ratio
        self._decay = np.log(2) / max(transition_halflife, 1)

        self._current_regime: Optional[Regime] = None
        self._prev_regime: Optional[Regime] = None
        self._transition_day: int = 0
        self._state_history: List[HedgeState] = []
        self._transitions: List[RegimeTransition] = []

    # ------------------------------------------------------------------
    # Parameter blending
    # ------------------------------------------------------------------

    @staticmethod
    def blend_params(
        old: HedgeParams, new: HedgeParams, alpha: float,
    ) -> HedgeParams:
        """Exponentially blend two parameter sets.  alpha=1 → fully new."""
        a = max(0.0, min(1.0, alpha))
        return HedgeParams(
            vix_floor=old.vix_floor * (1 - a) + new.vix_floor * a,
            vix_ceiling=old.vix_ceiling * (1 - a) + new.vix_ceiling * a,
            stop_multiplier=old.stop_multiplier * (1 - a) + new.stop_multiplier * a,
            hv_scale=old.hv_scale * (1 - a) + new.hv_scale * a,
            hedge_ratio=old.hedge_ratio * (1 - a) + new.hedge_ratio * a,
            cost_budget=old.cost_budget * (1 - a) + new.cost_budget * a,
        )

    def _compute_alpha(self, days_since_transition: int) -> float:
        """Exponential blending weight: 1 - exp(-decay * t)."""
        return 1.0 - np.exp(-self._decay * days_since_transition)

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        regime: Regime,
        date: Optional[datetime] = None,
        daily_return: float = 0.0,
    ) -> HedgeState:
        """Process one day: detect transition, blend params, track cost."""
        dt = date or datetime.now()
        new_profile = self.profiles.get(regime, DEFAULT_PROFILES.get(regime, HedgeParams()))

        if regime != self._current_regime:
            self._prev_regime = self._current_regime
            if self._current_regime is not None:
                self._transitions.append(RegimeTransition(
                    date=dt,
                    from_regime=self._current_regime,
                    to_regime=regime,
                    transition_days=self.transition_halflife,
                ))
            self._current_regime = regime
            self._transition_day = 0

        # Blend if transitioning
        blended = False
        alpha = 1.0
        if self._prev_regime is not None and self._transition_day < self.transition_halflife * 3:
            alpha = self._compute_alpha(self._transition_day)
            old_profile = self.profiles.get(self._prev_regime, HedgeParams())
            params = self.blend_params(old_profile, new_profile, alpha)
            blended = alpha < 0.95
        else:
            params = new_profile

        self._transition_day += 1

        # Daily hedge cost estimate: hedge_ratio × cost_budget / 252
        daily_cost = params.hedge_ratio * params.cost_budget / TRADING_DAYS

        state = HedgeState(
            date=dt, regime=regime, params=params,
            blended=blended, blend_alpha=alpha,
            daily_cost=daily_cost,
        )
        self._state_history.append(state)
        return state

    def update_series(
        self,
        regimes: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> List[HedgeState]:
        """Process a full regime series."""
        results: List[HedgeState] = []
        for i, (dt, reg) in enumerate(regimes.items()):
            if not isinstance(reg, Regime):
                try:
                    reg = Regime(reg)
                except ValueError:
                    continue
            daily_ret = float(returns.iloc[i]) if returns is not None and i < len(returns) else 0.0
            results.append(self.update(reg, date=dt, daily_return=daily_ret))
        return results

    # ------------------------------------------------------------------
    # Hedge cost tracking
    # ------------------------------------------------------------------

    def cost_by_regime(
        self,
        returns: Optional[pd.Series] = None,
    ) -> List[HedgeCostSummary]:
        """Aggregate hedge costs per regime."""
        if not self._state_history:
            return []

        by_regime: Dict[str, List[HedgeState]] = {}
        for s in self._state_history:
            by_regime.setdefault(s.regime.value, []).append(s)

        results: List[HedgeCostSummary] = []
        for regime, states in sorted(by_regime.items()):
            costs = [s.daily_cost for s in states]
            total = sum(costs)
            avg = total / len(costs) if costs else 0.0
            mx = max(costs) if costs else 0.0

            # Cost as % of return for this regime
            regime_return = 0.0
            if returns is not None:
                regime_dates = {s.date for s in states}
                regime_ret = returns.loc[returns.index.isin(regime_dates)]
                regime_return = float(regime_ret.sum()) if not regime_ret.empty else 0.0

            cost_pct = total / abs(regime_return) if abs(regime_return) > 1e-8 else 0.0

            results.append(HedgeCostSummary(
                regime=regime, n_days=len(states),
                total_cost=total, avg_daily_cost=avg,
                max_daily_cost=mx, cost_as_pct_return=cost_pct,
            ))
        return results

    # ------------------------------------------------------------------
    # Backtest: adaptive vs static
    # ------------------------------------------------------------------

    def backtest(
        self,
        regimes: pd.Series,
        returns: pd.Series,
    ) -> BacktestComparison:
        """Compare adaptive hedging vs static hedging."""
        self.reset()
        states = self.update_series(regimes, returns)

        n = min(len(states), len(returns))
        adaptive_rets = np.zeros(n)
        static_rets = np.zeros(n)

        for i in range(n):
            r = float(returns.iloc[i])
            hr = states[i].params.hedge_ratio
            cost = states[i].daily_cost
            # Adaptive: hedge reduces losses, costs on gains
            adaptive_rets[i] = r * (1 - hr) - cost
            # Static
            static_rets[i] = r * (1 - self.static_hedge_ratio) - self.static_hedge_ratio * 0.005 / TRADING_DAYS

        def _metrics(rets):
            eq = np.cumprod(1 + rets)
            hwm = np.maximum.accumulate(eq)
            dd = 1 - eq / hwm
            pnl = float(eq[-1] - 1)
            max_dd = float(dd.max())
            mu = float(rets.mean())
            std = float(rets.std())
            sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
            return pnl, max_dd, sharpe

        a_pnl, a_dd, a_sh = _metrics(adaptive_rets)
        s_pnl, s_dd, s_sh = _metrics(static_rets)

        a_cost = sum(s.daily_cost for s in states[:n])
        s_cost = n * self.static_hedge_ratio * 0.005 / TRADING_DAYS

        return BacktestComparison(
            adaptive_pnl=a_pnl, static_pnl=s_pnl,
            adaptive_max_dd=a_dd, static_max_dd=s_dd,
            adaptive_sharpe=a_sh, static_sharpe=s_sh,
            adaptive_cost=a_cost, static_cost=s_cost,
            improvement_pnl=a_pnl - s_pnl,
            improvement_dd=s_dd - a_dd,
        )

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------

    def grid_sweep(
        self,
        regime: Regime,
        regimes: pd.Series,
        returns: pd.Series,
        hedge_ratios: Optional[List[float]] = None,
        stop_mults: Optional[List[float]] = None,
    ) -> GridSweepResult:
        """Search optimal hedge params for one regime via grid sweep."""
        hedge_ratios = hedge_ratios or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        stop_mults = stop_mults or [1.0, 1.5, 2.0, 2.5, 3.0]

        all_results: List[Dict] = []
        best_sharpe = -999.0
        best_hr = 0.0
        best_sm = 2.0

        for hr in hedge_ratios:
            for sm in stop_mults:
                # Build custom profiles with this regime's params overridden
                custom = dict(self.profiles)
                old = custom.get(regime, HedgeParams())
                custom[regime] = HedgeParams(
                    vix_floor=old.vix_floor, vix_ceiling=old.vix_ceiling,
                    stop_multiplier=sm, hv_scale=old.hv_scale,
                    hedge_ratio=hr, cost_budget=old.cost_budget,
                )
                eng = RegimeHedgeEngine(
                    profiles=custom,
                    transition_halflife=self.transition_halflife,
                    static_hedge_ratio=self.static_hedge_ratio,
                )
                comp = eng.backtest(regimes, returns)
                all_results.append({
                    "hedge_ratio": hr, "stop_mult": sm,
                    "sharpe": comp.adaptive_sharpe,
                    "pnl": comp.adaptive_pnl,
                    "max_dd": comp.adaptive_max_dd,
                })
                if comp.adaptive_sharpe > best_sharpe:
                    best_sharpe = comp.adaptive_sharpe
                    best_hr = hr
                    best_sm = sm

        return GridSweepResult(
            regime=regime.value, best_hedge_ratio=best_hr,
            best_stop_mult=best_sm, best_sharpe=best_sharpe,
            all_results=all_results,
        )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state_history(self) -> List[HedgeState]:
        return list(self._state_history)

    @property
    def transitions(self) -> List[RegimeTransition]:
        return list(self._transitions)

    def reset(self) -> None:
        self._current_regime = None
        self._prev_regime = None
        self._transition_day = 0
        self._state_history.clear()
        self._transitions.clear()

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 720, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        vmin, vmax = min(values), max(values)
        if vmax <= vmin:
            vmax = vmin + 0.01
        pad_l, pad_r, pad_t, pad_b = 50, 15, 28, 25
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        def tx(i): return pad_l + i / max(n - 1, 1) * pw
        def ty(v): return pad_t + (1 - (v - vmin) / (vmax - vmin)) * ph
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                      for i, v in enumerate(values))
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_regime_bar(
        states: List[HedgeState], width: int = 720, height: int = 45,
    ) -> str:
        if not states:
            return ""
        n = len(states)
        colors = {
            Regime.BULL: "#27ae60", Regime.BEAR: "#e74c3c",
            Regime.HIGH_VOL: "#e67e22", Regime.LOW_VOL: "#2980b9",
            Regime.CRASH: "#8e44ad",
        }
        bw = width / max(n, 1)
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        for i, s in enumerate(states):
            c = colors.get(s.regime, "#999")
            p.append(f'<rect x="{i * bw:.1f}" y="0" width="{bw + .5:.1f}" '
                     f'height="{height - 16}" fill="{c}"/>')
        lx = 5
        for r, c in colors.items():
            p.append(f'<rect x="{lx}" y="{height - 12}" width="8" height="8" fill="{c}"/>')
            p.append(f'<text x="{lx + 11}" y="{height - 4}" font-size="8" fill="#333">{r.value}</text>')
            lx += 65
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        comparison: Optional[BacktestComparison] = None,
        sweep: Optional[GridSweepResult] = None,
        returns: Optional[pd.Series] = None,
        output_path: str = "reports/regime_hedge.html",
    ) -> str:
        """HTML report: regime timeline, param evolution, cost comparison."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Regime timeline
        regime_svg = self._svg_regime_bar(self._state_history)

        # Hedge ratio evolution
        hr_vals = [s.params.hedge_ratio for s in self._state_history]
        hr_svg = self._svg_line(hr_vals, "Hedge Ratio Evolution", color="#e67e22")

        # Stop multiplier evolution
        sm_vals = [s.params.stop_multiplier for s in self._state_history]
        sm_svg = self._svg_line(sm_vals, "Stop Multiplier Evolution", color="#2980b9")

        # Daily cost evolution
        cost_vals = [s.daily_cost for s in self._state_history]
        cost_svg = self._svg_line(cost_vals, "Daily Hedge Cost", color="#e74c3c")

        # Cost summary table
        cost_summary = self.cost_by_regime(returns)
        cost_rows = [
            f"<tr><td>{c.regime}</td><td>{c.n_days}</td>"
            f"<td>{c.total_cost:.4f}</td><td>{c.avg_daily_cost:.6f}</td>"
            f"<td>{c.max_daily_cost:.6f}</td>"
            f"<td>{c.cost_as_pct_return:.1%}</td></tr>"
            for c in cost_summary
        ]

        # Comparison
        comp_html = ""
        if comparison:
            c = comparison
            comp_html = f"""
<h2>Adaptive vs Static Hedging</h2>
<table class="m"><tr><th></th><th>Adaptive</th><th>Static</th><th>Diff</th></tr>
<tr><td>P&amp;L</td><td>{c.adaptive_pnl:+.2%}</td><td>{c.static_pnl:+.2%}</td>
<td>{c.improvement_pnl:+.2%}</td></tr>
<tr><td>Max DD</td><td>{c.adaptive_max_dd:.2%}</td><td>{c.static_max_dd:.2%}</td>
<td>{c.improvement_dd:+.2%}</td></tr>
<tr><td>Sharpe</td><td>{c.adaptive_sharpe:.2f}</td><td>{c.static_sharpe:.2f}</td>
<td>{c.adaptive_sharpe - c.static_sharpe:+.2f}</td></tr>
<tr><td>Total Cost</td><td>{c.adaptive_cost:.4f}</td><td>{c.static_cost:.4f}</td>
<td>{c.adaptive_cost - c.static_cost:+.4f}</td></tr></table>"""

        # Sweep
        sweep_html = ""
        if sweep:
            sweep_html = f"""
<h2>Grid Sweep: {sweep.regime}</h2>
<p>Best hedge_ratio={sweep.best_hedge_ratio:.2f},
   stop_mult={sweep.best_stop_mult:.1f},
   Sharpe={sweep.best_sharpe:.2f}</p>"""

        # Transition log
        trans_rows = [
            f"<tr><td>{t.date.strftime('%Y-%m-%d') if hasattr(t.date, 'strftime') else t.date}</td>"
            f"<td>{t.from_regime.value}</td><td>{t.to_regime.value}</td>"
            f"<td>{t.transition_days}d</td></tr>"
            for t in self._transitions
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Regime Hedge Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.m {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Regime-Adaptive Hedge Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>States:</strong> {len(self._state_history)} |
   <strong>Transitions:</strong> {len(self._transitions)}</p>
</div>

<h2>Regime Timeline</h2>
{regime_svg}

<h2>Hedge Parameter Evolution</h2>
{hr_svg}
{sm_svg}
{cost_svg}

<h2>Cost by Regime</h2>
<table><tr><th>Regime</th><th>Days</th><th>Total Cost</th><th>Avg Daily</th>
<th>Max Daily</th><th>Cost/Return</th></tr>
{''.join(cost_rows)}</table>

{comp_html}
{sweep_html}

<h2>Regime Transitions</h2>
<table><tr><th>Date</th><th>From</th><th>To</th><th>Blend</th></tr>
{''.join(trans_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Regime hedge report -> %s", path)
        return str(path)
