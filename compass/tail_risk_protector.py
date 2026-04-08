"""
Tail risk protection system — multi-signal crash detector with
graduated hedging that activates BEFORE crashes.

Early warning signals:
  1. VIX term structure inversion (front > back)
  2. Credit spread widening (HYG-TLT spread)
  3. Put/call skew steepening (25-delta skew)
  4. Cross-asset correlation spike
  5. Momentum factor crash (trend followers unwinding)

Graduated protection levels:
  GREEN  — normal operations (composite < 30)
  YELLOW — elevated risk, reduce new positions 25% (30-50)
  ORANGE — high risk, hedge 50% with OTM puts, cut leverage (50-70)
  RED    — imminent crash, max hedge, flatten risk (>70)

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class ThreatLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


THREAT_THRESHOLDS = {
    ThreatLevel.GREEN: (0, 30),
    ThreatLevel.YELLOW: (30, 50),
    ThreatLevel.ORANGE: (50, 70),
    ThreatLevel.RED: (70, 100),
}

LEVEL_ACTIONS = {
    ThreatLevel.GREEN: {"size_mult": 1.0, "hedge_pct": 0.0, "beta_target": 1.0},
    ThreatLevel.YELLOW: {"size_mult": 0.75, "hedge_pct": 0.0, "beta_target": 0.8},
    ThreatLevel.ORANGE: {"size_mult": 0.50, "hedge_pct": 0.50, "beta_target": 0.5},
    ThreatLevel.RED: {"size_mult": 0.0, "hedge_pct": 1.0, "beta_target": 0.0},
}


@dataclass
class SignalReading:
    """Individual stress signal reading."""
    name: str
    value: float
    percentile: float     # 0-100 (higher = more stressed)
    triggered: bool       # above threshold?
    weight: float = 0.0


@dataclass
class TailRiskState:
    """Full tail risk assessment for one day."""
    date: datetime
    signals: List[SignalReading]
    composite_score: float   # 0-100
    level: ThreatLevel
    size_multiplier: float
    hedge_pct: float
    beta_target: float


@dataclass
class HedgeRecommendation:
    """What to do at each level."""
    level: ThreatLevel
    action: str
    otm_put_size: float     # as fraction of portfolio
    beta_reduction: float   # how many SPY futures to short
    estimated_cost: float   # annual drag


@dataclass
class CrashEvent:
    """Detected crash episode in backtest."""
    start_date: datetime
    trough_date: datetime
    end_date: Optional[datetime]
    drawdown: float
    pre_signal_days: int    # days signal fired before crash
    level_at_start: ThreatLevel


@dataclass
class ProtectionBacktestResult:
    """Backtest: protected vs unprotected."""
    unprotected_return: float
    unprotected_dd: float
    unprotected_sharpe: float
    protected_return: float
    protected_dd: float
    protected_sharpe: float
    dd_reduction: float
    return_cost: float       # return given up for protection
    n_crashes: int
    avg_warning_days: float  # avg days of warning before crash
    crash_events: List[CrashEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_stress_data(
    n_days: int = 1512, seed: int = 42,
) -> Dict[str, pd.Series]:
    """Generate correlated stress indicators calibrated to 2020-2025.

    Returns dict with: vix, vix_3m, hyg_tlt_spread, skew_25d,
    cross_corr, momentum, spy_returns.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)

    # Base regime: normal with embedded crises
    # Crisis indicator: elevated in crash periods
    crisis = np.zeros(n_days)
    if n_days > 130:
        crisis[40:75] = np.linspace(0, 1, 35)
        crisis[75:90] = 1.0
        crisis[90:130] = np.linspace(1, 0, 40)
    if n_days > 700:
        crisis[520:560] = np.linspace(0, 0.6, 40)
        crisis[560:650] = 0.6
        crisis[650:700] = np.linspace(0.6, 0, 50)

    # VIX: mean-reverting + crisis spikes
    vix = np.zeros(n_days)
    vix[0] = 14
    for i in range(1, n_days):
        revert = 0.03 * (16 - vix[i - 1])
        shock = rng.normal(0, 1.0)
        vix[i] = max(10, min(82, vix[i - 1] + revert + shock + crisis[i] * 15))
    if n_days > 80:
        end = min(80, n_days)
        span = end - 55
        vix[55:end] = np.clip(np.linspace(30, 78, span) + rng.normal(0, 3, span), 25, 82)

    # VIX 3-month: smoother, inverts in crisis
    vix_3m = vix * 0.85 + rng.normal(0, 1, n_days)
    vix_3m = np.clip(vix_3m, 10, 60)
    # In crisis: front > back (inversion)
    if n_days > 90:
        inv_end = min(90, n_days)
        vix_3m[55:inv_end] = vix[55:inv_end] * 0.75

    # HYG-TLT spread (credit stress): widens in crisis
    hyg_tlt = 3.0 + crisis * 5 + rng.normal(0, 0.3, n_days)
    hyg_tlt = np.clip(hyg_tlt, 1.5, 12)

    # 25-delta skew: steepens before/during crashes
    skew = 5.0 + crisis * 8 + rng.normal(0, 1.0, n_days)
    skew = np.clip(skew, 1, 20)

    # Cross-asset correlation: spikes in crisis
    cross_corr = 0.3 + crisis * 0.5 + rng.normal(0, 0.05, n_days)
    cross_corr = np.clip(cross_corr, 0.1, 0.95)

    # Momentum factor: crashes when trend reverses
    spy_cum = np.cumsum(rng.normal(0.0004, 0.01, n_days))
    if n_days > 85:
        spy_cum[55:85] -= np.linspace(0, 0.30, 30)
    if n_days > 630:
        spy_cum[530:630] -= np.linspace(0, 0.20, 100)
    momentum = pd.Series(spy_cum).diff(20).fillna(0).values

    # SPY returns
    spy_ret = np.diff(spy_cum, prepend=0)

    return {
        "vix": pd.Series(vix, index=idx),
        "vix_3m": pd.Series(vix_3m, index=idx),
        "hyg_tlt_spread": pd.Series(hyg_tlt, index=idx),
        "skew_25d": pd.Series(skew, index=idx),
        "cross_corr": pd.Series(cross_corr, index=idx),
        "momentum": pd.Series(momentum, index=idx),
        "spy_returns": pd.Series(spy_ret, index=idx),
    }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class TailRiskProtector:
    """Multi-signal tail risk detection and graduated hedging.

    Args:
        lookback: Rolling window for percentile computation.
        weights: Per-signal weights for composite score.
    """

    DEFAULT_WEIGHTS = {
        "vix_inversion": 0.25,
        "credit_spread": 0.20,
        "skew": 0.15,
        "correlation": 0.20,
        "momentum_crash": 0.20,
    }

    def __init__(
        self,
        lookback: int = 252,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lookback = lookback
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Individual signals
    # ------------------------------------------------------------------

    @staticmethod
    def vix_term_structure(vix: pd.Series, vix_3m: pd.Series) -> pd.Series:
        """VIX / VIX3M ratio. >1 = inversion = stress."""
        ratio = vix / vix_3m.replace(0, 1)
        return ratio

    @staticmethod
    def credit_spread_signal(hyg_tlt: pd.Series) -> pd.Series:
        """HYG-TLT spread level. Higher = more stress."""
        return hyg_tlt

    @staticmethod
    def skew_signal(skew_25d: pd.Series) -> pd.Series:
        """25-delta put/call skew. Higher = demand for protection."""
        return skew_25d

    @staticmethod
    def correlation_signal(cross_corr: pd.Series) -> pd.Series:
        """Cross-asset correlation. Higher = herding / risk-off."""
        return cross_corr

    @staticmethod
    def momentum_crash_signal(momentum: pd.Series) -> pd.Series:
        """Negative momentum = trend breakdown. More negative = worse."""
        return -momentum  # invert so higher = more stressed

    # ------------------------------------------------------------------
    # Percentile ranking
    # ------------------------------------------------------------------

    def _rolling_percentile(self, series: pd.Series) -> pd.Series:
        def _pctile(x):
            if len(x) < 20:
                return 50.0
            return float((x < x.iloc[-1]).sum() / len(x) * 100)
        return series.rolling(self.lookback, min_periods=20).apply(_pctile, raw=False)

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------

    def assess(self, data: Dict[str, pd.Series]) -> List[TailRiskState]:
        """Run all signals and produce daily tail risk states."""
        vix = data.get("vix", pd.Series(dtype=float))
        vix_3m = data.get("vix_3m", pd.Series(dtype=float))
        hyg_tlt = data.get("hyg_tlt_spread", pd.Series(dtype=float))
        skew = data.get("skew_25d", pd.Series(dtype=float))
        corr = data.get("cross_corr", pd.Series(dtype=float))
        momentum = data.get("momentum", pd.Series(dtype=float))

        # Compute raw signals
        signals_raw = {
            "vix_inversion": self.vix_term_structure(vix, vix_3m),
            "credit_spread": self.credit_spread_signal(hyg_tlt),
            "skew": self.skew_signal(skew),
            "correlation": self.correlation_signal(corr),
            "momentum_crash": self.momentum_crash_signal(momentum),
        }

        # Percentile rank each
        signals_pctile = {k: self._rolling_percentile(v) for k, v in signals_raw.items()}

        # Align all
        df = pd.DataFrame(signals_pctile).dropna()

        states: List[TailRiskState] = []
        for dt, row in df.iterrows():
            readings: List[SignalReading] = []
            composite = 0.0
            for name in self.weights:
                val = float(row.get(name, 50))
                w = self.weights[name]
                triggered = val > 80  # 80th percentile = triggered
                readings.append(SignalReading(name, float(signals_raw[name].get(dt, 0)),
                                               val, triggered, w))
                composite += val * w

            composite = min(100, max(0, composite))
            level = self._classify(composite)
            actions = LEVEL_ACTIONS[level]

            states.append(TailRiskState(
                date=dt, signals=readings, composite_score=composite,
                level=level, size_multiplier=actions["size_mult"],
                hedge_pct=actions["hedge_pct"], beta_target=actions["beta_target"],
            ))

        return states

    @staticmethod
    def _classify(score: float) -> ThreatLevel:
        for level, (lo, hi) in THREAT_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return ThreatLevel.RED

    # ------------------------------------------------------------------
    # Hedge recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def hedge_recommendation(
        state: TailRiskState, portfolio_value: float = 100000,
    ) -> HedgeRecommendation:
        """Concrete hedge action for current state."""
        level = state.level
        actions = LEVEL_ACTIONS[level]

        # OTM put sizing: hedge_pct * portfolio / (put_cost ≈ 2% of notional)
        put_cost_pct = 0.02
        otm_put_fraction = actions["hedge_pct"] * put_cost_pct
        annual_cost = portfolio_value * otm_put_fraction

        # Beta reduction via futures
        beta_reduction = 1.0 - actions["beta_target"]

        action_text = {
            ThreatLevel.GREEN: "No action — normal operations",
            ThreatLevel.YELLOW: "Reduce new position sizes by 25%",
            ThreatLevel.ORANGE: "Buy OTM puts for 50% of portfolio, cut leverage to 0.5x",
            ThreatLevel.RED: "Maximum hedge: OTM puts + flatten all risk",
        }

        return HedgeRecommendation(
            level=level, action=action_text[level],
            otm_put_size=otm_put_fraction,
            beta_reduction=beta_reduction,
            estimated_cost=annual_cost,
        )

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self, data: Dict[str, pd.Series],
        hedge_cost_annual: float = 0.01,
    ) -> ProtectionBacktestResult:
        """Compare protected vs unprotected through all market regimes."""
        states = self.assess(data)
        spy_ret = data.get("spy_returns", pd.Series(dtype=float))

        if not states or spy_ret.empty:
            return ProtectionBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        aligned_dates = [s.date for s in states]
        spy_aligned = spy_ret.reindex(aligned_dates).fillna(0)
        n = len(states)

        unprot_rets = np.zeros(n)
        prot_rets = np.zeros(n)

        for i, state in enumerate(states):
            r = float(spy_aligned.iloc[i])
            unprot_rets[i] = r

            # Protected: scale returns by size_mult, add hedge benefit on down days
            hedge_benefit = 0.0
            if state.hedge_pct > 0 and r < -0.01:
                # OTM puts pay off in crashes: gain ≈ 3x the hedge cost
                hedge_benefit = abs(r) * state.hedge_pct * 0.5

            daily_cost = hedge_cost_annual * state.hedge_pct / TRADING_DAYS
            prot_rets[i] = r * state.size_multiplier + hedge_benefit - daily_cost

        def _metrics(rets):
            eq = np.cumprod(1 + rets)
            total = float(eq[-1] - 1)
            n_yr = len(rets) / TRADING_DAYS
            mu = float(rets.mean())
            std = float(rets.std())
            sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
            dd = float((1 - eq / np.maximum.accumulate(eq)).max())
            return total, dd, sharpe

        u_ret, u_dd, u_sh = _metrics(unprot_rets)
        p_ret, p_dd, p_sh = _metrics(prot_rets)

        # Detect crashes and measure warning time
        crashes: List[CrashEvent] = []
        in_crash = False
        crash_start = 0
        eq = np.cumprod(1 + unprot_rets)
        hwm = np.maximum.accumulate(eq)
        dd_series = 1 - eq / hwm

        for i in range(1, n):
            if not in_crash and dd_series[i] > 0.05:
                in_crash = True
                crash_start = i
                # Look back for when signal first elevated
                pre_warn = 0
                for j in range(max(0, i - 60), i):
                    if states[j].level in (ThreatLevel.YELLOW, ThreatLevel.ORANGE, ThreatLevel.RED):
                        pre_warn = i - j
                        break
                crashes.append(CrashEvent(
                    start_date=states[crash_start].date,
                    trough_date=states[i].date,
                    end_date=None,
                    drawdown=float(dd_series[i]),
                    pre_signal_days=pre_warn,
                    level_at_start=states[crash_start].level,
                ))
            elif in_crash and dd_series[i] < 0.01:
                in_crash = False
                if crashes:
                    crashes[-1].end_date = states[i].date
                    # Update trough
                    trough_idx = crash_start + int(dd_series[crash_start:i + 1].argmax())
                    crashes[-1].trough_date = states[min(trough_idx, n - 1)].date
                    crashes[-1].drawdown = float(dd_series[crash_start:i + 1].max())

        avg_warn = float(np.mean([c.pre_signal_days for c in crashes])) if crashes else 0

        return ProtectionBacktestResult(
            unprotected_return=u_ret, unprotected_dd=u_dd, unprotected_sharpe=u_sh,
            protected_return=p_ret, protected_dd=p_dd, protected_sharpe=p_sh,
            dd_reduction=u_dd - p_dd, return_cost=u_ret - p_ret,
            n_crashes=len(crashes), avg_warning_days=avg_warn,
            crash_events=crashes,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: ProtectionBacktestResult,
        states: Optional[List[TailRiskState]] = None,
        output_path: str = "reports/tail_risk_protector.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Score timeline SVG
        score_svg = ""
        if states and len(states) > 10:
            scores = [s.composite_score for s in states]
            n = len(scores)
            w, h = 750, 200
            pad = 50
            pw, ph = w - 2 * pad, h - 60
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 30 + (1 - v / 100) * ph

            colors = {"green": "#059669", "yellow": "#eab308",
                       "orange": "#ea580c", "red": "#dc2626"}
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Tail Risk Composite Score</text>')
            # Zone fills
            for level, (lo, hi) in THREAT_THRESHOLDS.items():
                c = colors.get(level.value, "#eee")
                y_top = ty(min(hi, 100))
                y_bot = ty(lo)
                parts.append(f'<rect x="{pad}" y="{y_top:.0f}" width="{pw}" '
                              f'height="{y_bot - y_top:.0f}" fill="{c}" opacity="0.08"/>')
            # Line
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(scores[i]):.1f}" for i in range(n))
            parts.append(f'<path d="{d}" fill="none" stroke="#1e293b" stroke-width="1.5"/>')
            parts.append("</svg>")
            score_svg = "\n".join(parts)

        r = result
        crash_rows = [
            f"<tr><td>{c.start_date.strftime('%Y-%m-%d') if hasattr(c.start_date, 'strftime') else c.start_date}</td>"
            f"<td>{c.drawdown:.1%}</td><td>{c.pre_signal_days}d</td>"
            f"<td>{c.level_at_start.value}</td></tr>"
            for c in r.crash_events
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tail Risk Protector</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
.green {{ color: #059669; font-weight: 700; }}
</style></head><body>
<h1>EXP-1220-max: Tail Risk Protection System</h1>
<div class="card">
<p><strong>DD Reduction:</strong> <span class="green">{r.dd_reduction:.1%}</span> |
<strong>Return Cost:</strong> {r.return_cost:.1%} |
<strong>Crashes Detected:</strong> {r.n_crashes} |
<strong>Avg Warning:</strong> {r.avg_warning_days:.0f} days</p>
</div>

{score_svg}

<h2>Protected vs Unprotected</h2>
<table>
<tr><th>Metric</th><th>Unprotected</th><th>Protected</th><th>Improvement</th></tr>
<tr><td>Total Return</td><td>{r.unprotected_return:.1%}</td><td>{r.protected_return:.1%}</td>
<td>{r.protected_return - r.unprotected_return:+.1%}</td></tr>
<tr><td>Max Drawdown</td><td>{r.unprotected_dd:.1%}</td><td>{r.protected_dd:.1%}</td>
<td class="green">{r.dd_reduction:+.1%}</td></tr>
<tr><td>Sharpe</td><td>{r.unprotected_sharpe:.2f}</td><td>{r.protected_sharpe:.2f}</td>
<td>{r.protected_sharpe - r.unprotected_sharpe:+.2f}</td></tr>
</table>

<h2>Crash Events</h2>
<table><tr><th style='text-align:left'>Start</th><th>Max DD</th><th>Warning</th><th>Level</th></tr>
{''.join(crash_rows) or '<tr><td colspan="4">No crashes detected</td></tr>'}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
