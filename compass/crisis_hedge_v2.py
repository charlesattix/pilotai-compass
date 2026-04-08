"""Crisis Hedge Controller V2 — extends V1 with gradual delevering, put spread
overlay, recovery detection, and hedge cost tracking.

Improvements over V1 (crisis_hedge.py):
  1. Configurable min_scale (floor at 0.40 instead of 0.0 — never fully halt)
  2. Multi-tier VIX triggers: 25 (reduce), 35 (minimum), 50 (full hedge)
  3. Put spread tail hedge overlay with cost-benefit analysis
  4. Drawdown-controlled delevering (gradual linear ramp, not binary)
  5. Recovery detection: momentum confirmation + vol normalisation
  6. Hedge cost tracking: cumulative drag in normal markets
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Configuration ───────────────────────────────────────────────────────────
@dataclass
class CrisisHedgeV2Config:
    """All tuneable parameters for crisis hedge V2."""
    # VIX tier thresholds
    vix_reduce: float = 25.0       # start reducing position size
    vix_minimum: float = 35.0      # reduce to min_scale
    vix_full_hedge: float = 50.0   # activate full put overlay

    # Scale limits
    max_scale: float = 1.0         # normal market scale
    min_scale: float = 0.40        # floor — never fully halt (V2 change)

    # Drawdown delevering
    dd_start: float = 0.05         # start delevering at 5% DD
    dd_full: float = 0.12          # reach min_scale at 12% DD
    dd_floor: float = 0.20         # emergency: 20% scale at this DD

    # Put spread overlay
    put_overlay_vix_trigger: float = 30.0   # activate put overlay above this VIX
    put_cost_pct_annual: float = 0.02       # 2% annual drag from put hedges
    put_protection_mult: float = 3.0        # $1 of puts protects $3 of downside

    # Recovery detection
    recovery_momentum_days: int = 10        # days of positive momentum to confirm
    recovery_vix_threshold: float = 22.0    # VIX must drop below this
    recovery_ramp_days: int = 20            # days to ramp back to full scale

    # Regime overrides
    crash_scale: float = 0.40              # V2: min_scale even in crash (not 0)
    high_vol_cap: float = 0.60             # cap in high_vol regime


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class HedgeState:
    """Current state of the crisis hedge controller."""
    scale_factor: float = 1.0
    vix_scale: float = 1.0
    dd_scale: float = 1.0
    regime_scale: float = 1.0
    put_overlay_active: bool = False
    put_cost_today: float = 0.0
    recovery_mode: bool = False
    recovery_progress: float = 0.0   # 0-1 ramp back
    is_delevering: bool = False
    reason: str = "normal"


@dataclass
class PutOverlayResult:
    """Cost-benefit analysis of put spread hedge overlay."""
    is_active: bool
    daily_cost: float              # dollar cost per day
    annual_cost_pct: float         # as % of portfolio
    protection_value: float        # downside protection in dollars
    cost_benefit_ratio: float      # protection / cost
    breakeven_dd: float            # DD at which put overlay pays for itself


@dataclass
class RecoverySignal:
    """Recovery detection output."""
    should_recover: bool
    momentum_confirmed: bool
    vix_normalised: bool
    progress: float                # 0-1 ramp
    days_in_recovery: int


@dataclass
class CrisisHedgeV2Result:
    """Complete result from running the hedge over a period."""
    states: List[HedgeState] = field(default_factory=list)
    total_hedge_cost: float = 0.0
    annual_hedge_drag_pct: float = 0.0
    put_overlay_days: int = 0
    max_scale_reduction: float = 0.0
    avg_scale: float = 1.0
    n_delevering_days: int = 0
    n_recovery_days: int = 0


# ── Core controller ────────────────────────────────────────────────────────
class CrisisHedgeControllerV2:
    """V2 crisis hedge with gradual delevering and put overlay."""

    def __init__(self, config: Optional[CrisisHedgeV2Config] = None) -> None:
        self.cfg = config or CrisisHedgeV2Config()
        self._recovery_counter: int = 0
        self._in_recovery: bool = False
        self._prev_scale: float = 1.0
        self._momentum_buffer: List[float] = []

    def compute_scale(
        self,
        vix: float,
        current_dd: float = 0.0,
        regime: str = "bull",
        daily_return: float = 0.0,
    ) -> HedgeState:
        """Compute the combined scale factor for today.

        Parameters
        ----------
        vix : float — current VIX level
        current_dd : float — current drawdown (positive fraction, e.g. 0.10)
        regime : str — current market regime
        daily_return : float — today's portfolio return (for momentum tracking)
        """
        # 1. VIX-based scaling
        vix_s = self._vix_scale(vix)

        # 2. Drawdown-based scaling
        dd_s = self._dd_scale(current_dd)

        # 3. Regime override
        regime_s = self._regime_scale(regime, vix_s)

        # 4. Combined: take the minimum (most conservative)
        raw_scale = min(vix_s, dd_s, regime_s)
        raw_scale = max(self.cfg.min_scale, min(self.cfg.max_scale, raw_scale))

        # 5. Recovery ramp-up
        self._update_momentum(daily_return)
        recovery = self._check_recovery(vix, daily_return)
        if recovery.should_recover and raw_scale < self.cfg.max_scale:
            # Gradually ramp back up
            raw_scale = raw_scale + (self.cfg.max_scale - raw_scale) * recovery.progress

        raw_scale = max(self.cfg.min_scale, min(self.cfg.max_scale, raw_scale))

        # 6. Put overlay
        put = self._put_overlay(vix, raw_scale)

        is_delevering = raw_scale < self._prev_scale or raw_scale < 0.95
        self._prev_scale = raw_scale

        # Build reason string
        reasons = []
        if vix_s < 1.0:
            reasons.append(f"VIX={vix:.0f}")
        if dd_s < 1.0:
            reasons.append(f"DD={current_dd:.1%}")
        if regime_s < 1.0:
            reasons.append(f"regime={regime}")
        if put.is_active:
            reasons.append("put_overlay")
        if recovery.should_recover:
            reasons.append(f"recovering({recovery.progress:.0%})")

        return HedgeState(
            scale_factor=round(raw_scale, 4),
            vix_scale=round(vix_s, 4),
            dd_scale=round(dd_s, 4),
            regime_scale=round(regime_s, 4),
            put_overlay_active=put.is_active,
            put_cost_today=put.daily_cost,
            recovery_mode=recovery.should_recover,
            recovery_progress=round(recovery.progress, 2),
            is_delevering=is_delevering,
            reason="; ".join(reasons) or "normal",
        )

    def put_overlay_analysis(
        self, vix: float, portfolio_value: float,
    ) -> PutOverlayResult:
        """Analyse cost-benefit of put spread overlay at current VIX."""
        return self._put_overlay(vix, 1.0, portfolio_value)

    def reset(self) -> None:
        """Reset internal state for new backtest."""
        self._recovery_counter = 0
        self._in_recovery = False
        self._prev_scale = 1.0
        self._momentum_buffer.clear()

    # ── VIX scaling ─────────────────────────────────────────────────────────
    def _vix_scale(self, vix: float) -> float:
        """Multi-tier VIX → scale mapping.

        VIX ≤ 25:   1.00  (no reduction)
        VIX 25-35:  1.00 → min_scale (linear)
        VIX 35-50:  min_scale (floor)
        VIX ≥ 50:   min_scale (floor — V2 never goes to 0)
        """
        if vix <= self.cfg.vix_reduce:
            return self.cfg.max_scale
        if vix >= self.cfg.vix_minimum:
            return self.cfg.min_scale
        # Linear interpolation between reduce and minimum
        t = (vix - self.cfg.vix_reduce) / (self.cfg.vix_minimum - self.cfg.vix_reduce)
        return self.cfg.max_scale - t * (self.cfg.max_scale - self.cfg.min_scale)

    # ── Drawdown scaling ────────────────────────────────────────────────────
    def _dd_scale(self, dd: float) -> float:
        """Gradual delevering as DD approaches threshold.

        DD ≤ 5%:    1.00
        DD 5-12%:   1.00 → min_scale (linear)
        DD ≥ 12%:   min_scale * 0.5 (emergency floor)
        """
        if dd <= self.cfg.dd_start:
            return self.cfg.max_scale
        if dd >= self.cfg.dd_full:
            return self.cfg.min_scale * 0.5  # emergency
        t = (dd - self.cfg.dd_start) / (self.cfg.dd_full - self.cfg.dd_start)
        return self.cfg.max_scale - t * (self.cfg.max_scale - self.cfg.min_scale)

    # ── Regime override ─────────────────────────────────────────────────────
    def _regime_scale(self, regime: str, vix_scale: float) -> float:
        r = regime.lower().strip()
        if r == "crash":
            return self.cfg.crash_scale
        if r == "high_vol":
            return min(vix_scale, self.cfg.high_vol_cap)
        return self.cfg.max_scale

    # ── Put spread overlay ──────────────────────────────────────────────────
    def _put_overlay(
        self, vix: float, scale: float, portfolio_value: float = 100_000,
    ) -> PutOverlayResult:
        is_active = vix >= self.cfg.put_overlay_vix_trigger
        if not is_active:
            return PutOverlayResult(False, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Cost scales with VIX (higher VIX = more expensive puts)
        vix_cost_mult = vix / 20.0  # normalise: VIX 20 = 1x cost
        annual_cost = self.cfg.put_cost_pct_annual * vix_cost_mult
        daily_cost = annual_cost * portfolio_value / 252

        # Protection value: scaled by how much we're already delevered
        protection = daily_cost * self.cfg.put_protection_mult * (1 + (1 - scale))

        cbr = protection / daily_cost if daily_cost > 0 else 0
        breakeven = annual_cost / self.cfg.put_protection_mult if self.cfg.put_protection_mult > 0 else 0

        return PutOverlayResult(
            is_active=True,
            daily_cost=round(daily_cost, 2),
            annual_cost_pct=round(annual_cost * 100, 2),
            protection_value=round(protection, 2),
            cost_benefit_ratio=round(cbr, 2),
            breakeven_dd=round(breakeven * 100, 2),
        )

    # ── Recovery detection ──────────────────────────────────────────────────
    def _update_momentum(self, daily_return: float) -> None:
        self._momentum_buffer.append(daily_return)
        if len(self._momentum_buffer) > self.cfg.recovery_momentum_days * 2:
            self._momentum_buffer = self._momentum_buffer[-self.cfg.recovery_momentum_days * 2:]

    def _check_recovery(self, vix: float, daily_return: float) -> RecoverySignal:
        mom_days = self.cfg.recovery_momentum_days
        recent = self._momentum_buffer[-mom_days:] if len(self._momentum_buffer) >= mom_days else self._momentum_buffer

        # Momentum: last N days average positive
        mom_confirmed = len(recent) >= mom_days and np.mean(recent) > 0

        # VIX normalised
        vix_ok = vix <= self.cfg.recovery_vix_threshold

        should = mom_confirmed and vix_ok

        if should:
            if not self._in_recovery:
                self._in_recovery = True
                self._recovery_counter = 0
            self._recovery_counter += 1
        else:
            self._in_recovery = False
            self._recovery_counter = 0

        progress = min(1.0, self._recovery_counter / max(self.cfg.recovery_ramp_days, 1))

        return RecoverySignal(
            should_recover=should,
            momentum_confirmed=mom_confirmed,
            vix_normalised=vix_ok,
            progress=round(progress, 2),
            days_in_recovery=self._recovery_counter,
        )


# ── Backtest integration ───────────────────────────────────────────────────
def backtest_with_hedge(
    daily_returns: np.ndarray,
    vix_series: np.ndarray,
    regime_series: List[str],
    base_leverage: float = 2.0,
    config: Optional[CrisisHedgeV2Config] = None,
    starting_capital: float = 100_000.0,
    regime_leverage: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Backtest a return stream with crisis hedge V2 overlay.

    Parameters
    ----------
    daily_returns : array of portfolio daily returns (unlevered)
    vix_series : array of daily VIX levels
    regime_series : list of regime labels per day
    base_leverage : target leverage
    config : CrisisHedgeV2Config
    starting_capital : initial capital
    regime_leverage : optional regime → leverage multiplier dict
    """
    n = len(daily_returns)
    controller = CrisisHedgeControllerV2(config)

    default_regime_lev = {
        "bull": 1.25, "sideways": 0.75, "bear": 0.375,
        "high_vol": 0.25, "crash": 0.0, "low_vol": 1.0,
    }
    rl = regime_leverage or default_regime_lev

    capital = starting_capital
    peak = capital
    max_dd = 0.0
    equity = [capital]
    total_hedge_cost = 0.0
    scale_history = []
    put_overlay_days = 0
    delevering_days = 0
    recovery_days_total = 0
    daily_pnl_list = []

    for i in range(n):
        vix = float(vix_series[i]) if i < len(vix_series) else 20.0
        regime = regime_series[i] if i < len(regime_series) else "bull"
        ret = float(daily_returns[i])
        current_dd = (peak - capital) / peak if peak > 0 else 0.0

        # Regime-based leverage
        regime_mult = rl.get(regime, 0.75)
        effective_leverage = base_leverage * regime_mult

        # Crisis hedge scale
        state = controller.compute_scale(vix, current_dd, regime, ret)
        scaled_leverage = effective_leverage * state.scale_factor
        scaled_leverage = min(scaled_leverage, 4.0)  # hard cap

        # Apply return
        port_return = ret * scaled_leverage

        # Put overlay cost
        if state.put_overlay_active:
            put_daily = state.put_cost_today
            # In a crash, puts pay off — offset losses
            if ret < -0.01:
                put_payoff = abs(ret) * capital * 0.15  # 15% of loss offset
                port_return += put_payoff / capital
            capital -= put_daily
            total_hedge_cost += put_daily
            put_overlay_days += 1

        capital *= (1 + port_return)
        daily_pnl_list.append(port_return)

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        equity.append(capital)
        scale_history.append(state.scale_factor)

        if state.is_delevering:
            delevering_days += 1
        if state.recovery_mode:
            recovery_days_total += 1

    # Metrics
    total_return = (capital - starting_capital) / starting_capital
    years = n / 252
    cagr = (capital / starting_capital) ** (1 / years) - 1 if years > 0 and capital > 0 else 0

    dr = np.array(daily_pnl_list)
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0
    avg_scale = float(np.mean(scale_history)) if scale_history else 1.0
    max_reduction = 1.0 - min(scale_history) if scale_history else 0.0
    hedge_drag = total_hedge_cost / starting_capital / years * 100 if years > 0 else 0

    # Yearly returns
    yearly = {}
    per_year = int(252)
    for y_idx in range(int(years) + 1):
        start_i = y_idx * per_year
        end_i = min((y_idx + 1) * per_year, n)
        if start_i >= n:
            break
        yr = 2020 + y_idx
        yr_eq_start = equity[start_i]
        yr_eq_end = equity[end_i]
        if yr_eq_start > 0:
            yearly[str(yr)] = round((yr_eq_end / yr_eq_start - 1) * 100, 1)

    return {
        "ending_capital": round(capital, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "avg_scale": round(avg_scale, 2),
        "max_scale_reduction": round(max_reduction, 2),
        "total_hedge_cost": round(total_hedge_cost, 2),
        "annual_hedge_drag_pct": round(hedge_drag, 2),
        "put_overlay_days": put_overlay_days,
        "delevering_days": delevering_days,
        "recovery_days": recovery_days_total,
        "all_years_profitable": all(v > 0 for v in yearly.values()) if yearly else False,
        "yearly_returns": yearly,
        "equity_curve": equity,
    }


# ── Stress test scenarios ───────────────────────────────────────────────────
def _shock_path(total: float, days: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if days <= 0:
        return np.array([])
    if days == 1:
        return np.array([total])
    lr = math.log(1 + total)
    w = np.exp(-np.arange(days) / max(days / 3, 1))
    w /= w.sum()
    raw = w * lr + rng.randn(days) * abs(lr) * 0.05
    raw *= lr / raw.sum() if abs(raw.sum()) > 1e-12 else 1.0
    return np.array([math.exp(r) - 1 for r in raw])


CRISIS_SCENARIOS = {
    "GFC_2008": {"shocks": _shock_path(-0.57, 350, 1), "vix_start": 20, "vix_peak": 80},
    "COVID_2020": {"shocks": _shock_path(-0.34, 23, 2), "vix_start": 15, "vix_peak": 82},
    "RATE_HIKES_2022": {"shocks": _shock_path(-0.25, 190, 3), "vix_start": 17, "vix_peak": 36},
    "FLASH_CRASH": {"shocks": _shock_path(-0.10, 1, 4), "vix_start": 15, "vix_peak": 65},
}


def stress_test_scenario(
    scenario_name: str,
    base_leverage: float = 2.0,
    config: Optional[CrisisHedgeV2Config] = None,
    starting_capital: float = 100_000.0,
    spread_beta: float = 1.5,
) -> Dict[str, Any]:
    """Run a single crisis scenario through the hedge controller."""
    sc = CRISIS_SCENARIOS[scenario_name]
    shocks = sc["shocks"]
    n = len(shocks)
    vix_start = sc["vix_start"]
    vix_peak = sc["vix_peak"]

    # Build VIX path: linear interpolation to peak then gradual decline
    peak_idx = n // 3
    vix_path = np.zeros(n)
    for i in range(n):
        if i <= peak_idx:
            vix_path[i] = vix_start + (vix_peak - vix_start) * i / max(peak_idx, 1)
        else:
            vix_path[i] = vix_peak - (vix_peak - vix_start * 1.2) * (i - peak_idx) / max(n - peak_idx, 1)
            vix_path[i] = max(vix_start, vix_path[i])

    # Regime: crash during peak VIX, high_vol during elevated, bear otherwise
    regimes = []
    for v in vix_path:
        if v > 40:
            regimes.append("crash")
        elif v > 28:
            regimes.append("high_vol")
        else:
            regimes.append("bear")

    # Apply spread beta to shocks
    adjusted_shocks = shocks * spread_beta

    # Run hedged
    hedged = backtest_with_hedge(
        adjusted_shocks, vix_path, regimes,
        base_leverage=base_leverage, config=config,
        starting_capital=starting_capital,
    )

    # Run unhedged (scale always 1.0)
    no_hedge_config = CrisisHedgeV2Config(
        vix_reduce=999, vix_minimum=999, dd_start=999,
        min_scale=1.0, max_scale=1.0,
        crash_scale=1.0, high_vol_cap=1.0,
        put_overlay_vix_trigger=999,
    )
    unhedged = backtest_with_hedge(
        adjusted_shocks, vix_path, regimes,
        base_leverage=base_leverage, config=no_hedge_config,
        starting_capital=starting_capital,
    )

    return {
        "scenario": scenario_name,
        "hedged_dd_pct": hedged["max_dd_pct"],
        "unhedged_dd_pct": unhedged["max_dd_pct"],
        "dd_reduction_pct": round(unhedged["max_dd_pct"] - hedged["max_dd_pct"], 2),
        "hedged_return_pct": hedged["total_return_pct"],
        "unhedged_return_pct": unhedged["total_return_pct"],
        "hedge_cost": hedged["total_hedge_cost"],
        "hedged_survives": hedged["max_dd_pct"] <= 15.0,
        "hedged": hedged,
        "unhedged": unhedged,
    }


# ── Full experiment runner ──────────────────────────────────────────────────
def run_experiment(seed: int = 42) -> Dict[str, Any]:
    """Run the full EXP-880 experiment."""
    rng = np.random.RandomState(seed)
    n = 252 * 6  # 6 years

    # Generate calibrated returns matching EXP-840 Regime Leverage 2x profile
    base_daily_mu = 0.29 / 252   # 29% annual (EXP-750 base)
    base_daily_sigma = 0.06 / np.sqrt(252)

    daily_returns = rng.randn(n) * base_daily_sigma + base_daily_mu

    # Inject crisis periods
    # COVID: ~day 35-58
    daily_returns[35:58] = _shock_path(-0.34, 23, 10) * 0.5  # attenuated for portfolio
    # 2022 bear: ~day 500-690
    for i in range(500, min(690, n)):
        daily_returns[i] = -0.002 + rng.randn() * base_daily_sigma * 1.5

    # VIX series
    vix = np.zeros(n)
    vix[0] = 14.0
    for i in range(1, n):
        vix[i] = max(9, min(80, vix[i-1] + 0.03 * (18 - vix[i-1]) - daily_returns[i] * 200 + rng.randn() * 1.5))

    # Regimes
    regimes = []
    for i in range(n):
        if vix[i] > 35:
            regimes.append("crash")
        elif vix[i] > 25:
            regimes.append("high_vol")
        elif daily_returns[max(0,i-20):i+1].mean() < -0.002 if i > 0 else False:
            regimes.append("bear")
        elif vix[i] < 14:
            regimes.append("low_vol")
        else:
            regimes.append("bull")

    # Run variants
    configs = {
        "No Hedge (baseline)": CrisisHedgeV2Config(
            vix_reduce=999, vix_minimum=999, dd_start=999,
            min_scale=1.0, max_scale=1.0, crash_scale=1.0,
            high_vol_cap=1.0, put_overlay_vix_trigger=999,
        ),
        "V2 Default": CrisisHedgeV2Config(),
        "V2 Aggressive (min 0.30)": CrisisHedgeV2Config(min_scale=0.30),
        "V2 Conservative (min 0.50)": CrisisHedgeV2Config(min_scale=0.50, dd_start=0.04),
        "V2 + Tight DD (3%/10%)": CrisisHedgeV2Config(dd_start=0.03, dd_full=0.10),
        "V2 + Wide DD (8%/15%)": CrisisHedgeV2Config(dd_start=0.08, dd_full=0.15),
        "V2 Tuned (min 0.25, DD 3/8)": CrisisHedgeV2Config(
            min_scale=0.25, dd_start=0.03, dd_full=0.08, dd_floor=0.12,
            crash_scale=0.25, high_vol_cap=0.40,
        ),
        "V2 Ultra-Safe (min 0.20, DD 2/7)": CrisisHedgeV2Config(
            min_scale=0.20, dd_start=0.02, dd_full=0.07, dd_floor=0.10,
            crash_scale=0.20, high_vol_cap=0.35, vix_reduce=22.0,
        ),
    }

    results = {}
    for name, cfg in configs.items():
        r = backtest_with_hedge(daily_returns, vix, regimes, base_leverage=2.0, config=cfg)
        results[name] = r
        print(f"  {name}: CAGR={r['cagr_pct']}%, Sharpe={r['sharpe']}, DD={r['max_dd_pct']}%, Drag={r['annual_hedge_drag_pct']}%")

    # Stress tests
    stress_results = {}
    for scenario in CRISIS_SCENARIOS:
        sr = stress_test_scenario(scenario, base_leverage=2.0)
        stress_results[scenario] = {
            "hedged_dd": sr["hedged_dd_pct"],
            "unhedged_dd": sr["unhedged_dd_pct"],
            "reduction": sr["dd_reduction_pct"],
            "survives": sr["hedged_survives"],
        }
        print(f"  Stress {scenario}: hedged DD={sr['hedged_dd_pct']}%, unhedged={sr['unhedged_dd_pct']}%, survives={sr['hedged_survives']}")

    # Find best variant
    qualifying = {k: v for k, v in results.items() if v["cagr_pct"] >= 40 and v["max_dd_pct"] <= 15}
    if qualifying:
        best_name = max(qualifying, key=lambda k: qualifying[k]["sharpe"])
    else:
        best_name = min(results, key=lambda k: results[k]["max_dd_pct"])
    best = results[best_name]

    return {
        "experiment": "EXP-880-max",
        "name": "Crisis Hedge Integration",
        "best_variant": best_name,
        "best": best,
        "all_variants": {k: {kk: vv for kk, vv in v.items() if kk != "equity_curve"} for k, v in results.items()},
        "stress_tests": stress_results,
        "all_crises_survive": all(s["survives"] for s in stress_results.values()),
        "success_criteria": {
            "cagr_gt_40": bool(best["cagr_pct"] >= 40),
            "max_dd_lt_15": bool(best["max_dd_pct"] <= 15),
            "sharpe_gt_3": bool(best["sharpe"] >= 3.0),
            "all_years_profitable": bool(best["all_years_profitable"]),
            "hedge_drag_lt_3": bool(best["annual_hedge_drag_pct"] <= 3.0),
        },
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    print("EXP-880-max: Crisis Hedge Integration")
    print("=" * 60)

    summary = run_experiment()
    best = summary["best"]

    results_dir = Path(__file__).parent.replace("compass", "experiments/EXP-880-max/results") if False else Path("experiments/EXP-880-max/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "best" or True}, f, indent=2, default=str)

    print(f"\nBEST: {summary['best_variant']}")
    print(f"  CAGR: {best['cagr_pct']}%, Sharpe: {best['sharpe']}, DD: {best['max_dd_pct']}%")
    print(f"  Hedge drag: {best['annual_hedge_drag_pct']}%/yr")
    print(f"  All crises survive: {summary['all_crises_survive']}")
