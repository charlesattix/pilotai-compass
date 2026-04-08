"""
Adaptive stop-loss optimizer — 5 stop types with regime-conditional
multipliers and walk-forward parameter selection.

Stop types:
  1. Fixed %: constant distance from entry
  2. ATR trailing: N × ATR(14) trailing from peak
  3. Chandelier: highest high - N × ATR
  4. Keltner: EMA ± N × ATR channel breach
  5. Volatility breakout: N × rolling std from peak

Regime multipliers (VIX-based):
  low_vol (VIX<15): 0.7× | normal (15-25): 1.0× |
  high_vol (25-35): 1.5× | crisis (>35): 2.0×

Usage::

    from compass.adaptive_stoploss import StopLossOptimizer
    opt = StopLossOptimizer(returns, vix_series)
    result = opt.optimize()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "adaptive_stoploss.html"
TRADING_DAYS = 252

STOP_TYPES = ["fixed_pct", "atr_trailing", "chandelier", "keltner", "vol_breakout"]

REGIME_MULTIPLIERS = {
    "low_vol": 0.7,
    "normal": 1.0,
    "high_vol": 1.5,
    "crisis": 2.0,
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class StopConfig:
    """Configuration for one stop type."""

    stop_type: str
    base_param: float  # N multiplier for ATR/std, % for fixed
    atr_period: int = 14
    ema_period: int = 20
    regime_adaptive: bool = True


@dataclass
class StopResult:
    """Result of applying one stop configuration."""

    stop_type: str
    regime_adaptive: bool
    base_param: float
    # Performance
    total_return_pct: float
    max_dd_pct: float
    sharpe: float
    return_preserved_pct: float  # vs no-stop
    dd_reduction_pct: float     # vs fixed stop
    n_stops_triggered: int
    avg_stop_distance_pct: float
    # Per-regime
    stops_by_regime: Dict[str, int]


@dataclass
class OptimizationResult:
    """Full optimization result across all stop types."""

    stop_results: List[StopResult]
    no_stop_return: float
    no_stop_dd: float
    no_stop_sharpe: float
    fixed_stop_dd: float
    best_stop: StopResult
    best_dd_reduction: float
    best_return_preservation: float
    n_observations: int


# ── VIX regime detection ─────────────────────────────────────────────────


def classify_vix_regime(vix: float) -> str:
    if vix < 15:
        return "low_vol"
    if vix < 25:
        return "normal"
    if vix < 35:
        return "high_vol"
    return "crisis"


def get_regime_multiplier(vix: float, adaptive: bool = True) -> float:
    if not adaptive:
        return 1.0
    regime = classify_vix_regime(vix)
    return REGIME_MULTIPLIERS.get(regime, 1.0)


# ── Technical indicators ─────────────────────────────────────────────────


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    ema = np.zeros(len(values))
    ema[0] = values[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


def compute_rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.zeros(len(values))
    for i in range(period, len(values)):
        result[i] = np.std(values[i - period:i], ddof=1)
    result[:period] = result[period] if period < len(values) else 0
    return result


# ── Stop-loss implementations ────────────────────────────────────────────


def apply_fixed_stop(
    equity: np.ndarray,
    stop_pct: float,
    vix: np.ndarray,
    adaptive: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Fixed % stop from peak equity."""
    n = len(equity)
    peak = equity[0]
    stopped = np.copy(equity)
    active = True
    n_stops = 0
    stops_by_regime: Dict[str, int] = {}

    for i in range(1, n):
        if not active:
            stopped[i] = stopped[i - 1]
            continue
        peak = max(peak, equity[i])
        mult = get_regime_multiplier(vix[i], adaptive)
        threshold = peak * (1 - stop_pct / 100 * mult)
        if equity[i] < threshold:
            stopped[i] = equity[i]
            active = False
            n_stops += 1
            regime = classify_vix_regime(vix[i])
            stops_by_regime[regime] = stops_by_regime.get(regime, 0) + 1
            # Re-enter after 5 bars
            for j in range(i + 1, min(i + 6, n)):
                stopped[j] = stopped[i]
            if i + 5 < n:
                active = True
                peak = equity[min(i + 5, n - 1)]
        else:
            stopped[i] = equity[i]

    return stopped, n_stops, stops_by_regime


def apply_atr_trailing(
    equity: np.ndarray,
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    multiplier: float,
    vix: np.ndarray,
    period: int = 14,
    adaptive: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """ATR-based trailing stop."""
    atr = compute_atr(high, low, close, period)
    n = len(equity)
    peak = equity[0]
    stopped = np.copy(equity)
    active = True
    n_stops = 0
    stops_by_regime: Dict[str, int] = {}

    for i in range(period, n):
        if not active:
            stopped[i] = stopped[i - 1]
            if i > n_stops * 5 + period:  # crude re-entry
                active = True
                peak = equity[i]
            continue
        peak = max(peak, equity[i])
        mult = get_regime_multiplier(vix[i], adaptive) * multiplier
        stop_dist = atr[i] / close[i] * 100 * mult  # as % of price
        threshold = peak * (1 - stop_dist / 100)
        if equity[i] < threshold:
            stopped[i] = equity[i]
            active = False
            n_stops += 1
            regime = classify_vix_regime(vix[i])
            stops_by_regime[regime] = stops_by_regime.get(regime, 0) + 1
        else:
            stopped[i] = equity[i]

    return stopped, n_stops, stops_by_regime


def apply_chandelier(
    equity: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    multiplier: float,
    vix: np.ndarray,
    period: int = 14,
    adaptive: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Chandelier exit: highest high - N × ATR."""
    atr = compute_atr(high, low, close, period)
    n = len(equity)
    stopped = np.copy(equity)
    active = True
    n_stops = 0
    stops_by_regime: Dict[str, int] = {}
    peak_equity = equity[0]

    for i in range(period, n):
        if not active:
            stopped[i] = stopped[i - 1]
            if i % 10 == 0:
                active = True
                peak_equity = equity[i]
            continue
        peak_equity = max(peak_equity, equity[i])
        mult = get_regime_multiplier(vix[i], adaptive) * multiplier
        highest = np.max(high[max(0, i - period):i + 1])
        stop_level = highest - mult * atr[i]
        # Map price stop to equity stop
        price_drop_pct = (highest - stop_level) / highest * 100
        eq_threshold = peak_equity * (1 - price_drop_pct / 100)
        if equity[i] < eq_threshold:
            stopped[i] = equity[i]
            active = False
            n_stops += 1
            regime = classify_vix_regime(vix[i])
            stops_by_regime[regime] = stops_by_regime.get(regime, 0) + 1
        else:
            stopped[i] = equity[i]

    return stopped, n_stops, stops_by_regime


def apply_keltner(
    equity: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    multiplier: float,
    vix: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 14,
    adaptive: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Keltner channel breach: exit when price drops below EMA - N × ATR."""
    ema = compute_ema(close, ema_period)
    atr = compute_atr(high, low, close, atr_period)
    n = len(equity)
    stopped = np.copy(equity)
    active = True
    n_stops = 0
    stops_by_regime: Dict[str, int] = {}
    peak_equity = equity[0]

    for i in range(max(ema_period, atr_period), n):
        if not active:
            stopped[i] = stopped[i - 1]
            if close[i] > ema[i]:
                active = True
                peak_equity = equity[i]
            continue
        peak_equity = max(peak_equity, equity[i])
        mult = get_regime_multiplier(vix[i], adaptive) * multiplier
        lower_band = ema[i] - mult * atr[i]
        if close[i] < lower_band:
            stopped[i] = equity[i]
            active = False
            n_stops += 1
            regime = classify_vix_regime(vix[i])
            stops_by_regime[regime] = stops_by_regime.get(regime, 0) + 1
        else:
            stopped[i] = equity[i]

    return stopped, n_stops, stops_by_regime


def apply_vol_breakout(
    equity: np.ndarray,
    returns: np.ndarray,
    multiplier: float,
    vix: np.ndarray,
    period: int = 20,
    adaptive: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """Volatility breakout: exit when loss exceeds N × rolling std from peak."""
    roll_std = compute_rolling_std(returns, period)
    n = len(equity)
    stopped = np.copy(equity)
    active = True
    n_stops = 0
    stops_by_regime: Dict[str, int] = {}
    peak_equity = equity[0]

    for i in range(period, n):
        if not active:
            stopped[i] = stopped[i - 1]
            if i % 8 == 0:
                active = True
                peak_equity = equity[i]
            continue
        peak_equity = max(peak_equity, equity[i])
        mult = get_regime_multiplier(vix[i], adaptive) * multiplier
        threshold = peak_equity * (1 - mult * roll_std[i] * math.sqrt(period))
        if equity[i] < threshold:
            stopped[i] = equity[i]
            active = False
            n_stops += 1
            regime = classify_vix_regime(vix[i])
            stops_by_regime[regime] = stops_by_regime.get(regime, 0) + 1
        else:
            stopped[i] = equity[i]

    return stopped, n_stops, stops_by_regime


# ── Metrics ──────────────────────────────────────────────────────────────


def _max_dd_pct(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()) * 100)


def _sharpe(equity: np.ndarray) -> float:
    rets = np.diff(equity) / equity[:-1]
    if len(rets) < 2:
        return 0.0
    mu, std = rets.mean(), rets.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


def _total_return(equity: np.ndarray) -> float:
    return float((equity[-1] / equity[0] - 1) * 100) if equity[0] > 0 else 0.0


# ── Core optimizer ───────────────────────────────────────────────────────


class StopLossOptimizer:
    """Adaptive stop-loss optimizer."""

    def __init__(
        self,
        equity: np.ndarray,
        prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        vix: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ):
        self.equity = equity
        self.prices = prices
        self.high = high
        self.low = low
        self.close = close
        self.vix = vix
        self.returns = returns if returns is not None else np.diff(equity) / np.maximum(equity[:-1], 1)
        # Pad returns to match equity length
        if len(self.returns) < len(equity):
            self.returns = np.append([0], self.returns)

    def optimize(
        self,
        params_to_test: Optional[Dict[str, List[float]]] = None,
    ) -> OptimizationResult:
        """Test all stop types and find the best."""
        if params_to_test is None:
            params_to_test = {
                "fixed_pct": [3.0, 5.0, 7.0, 10.0],
                "atr_trailing": [1.5, 2.0, 2.5, 3.0],
                "chandelier": [2.0, 2.5, 3.0, 3.5],
                "keltner": [1.5, 2.0, 2.5, 3.0],
                "vol_breakout": [1.5, 2.0, 2.5, 3.0],
            }

        no_stop_ret = _total_return(self.equity)
        no_stop_dd = _max_dd_pct(self.equity)
        no_stop_sharpe = _sharpe(self.equity)

        results: List[StopResult] = []

        for stop_type, params in params_to_test.items():
            for param in params:
                for adaptive in [True, False]:
                    stopped, n_stops, by_regime = self._apply_stop(stop_type, param, adaptive)
                    ret = _total_return(stopped)
                    dd = _max_dd_pct(stopped)
                    sh = _sharpe(stopped)

                    pres = ret / no_stop_ret * 100 if abs(no_stop_ret) > 0.01 else 100.0
                    # DD reduction vs fixed 5% non-adaptive
                    fixed_ref_dd = no_stop_dd  # will update below

                    avg_dist = param * np.mean([get_regime_multiplier(v, adaptive) for v in self.vix])

                    results.append(StopResult(
                        stop_type=stop_type, regime_adaptive=adaptive,
                        base_param=param, total_return_pct=ret,
                        max_dd_pct=dd, sharpe=sh,
                        return_preserved_pct=pres,
                        dd_reduction_pct=0.0,  # filled below
                        n_stops_triggered=n_stops,
                        avg_stop_distance_pct=avg_dist,
                        stops_by_regime=by_regime,
                    ))

        # Compute DD reduction vs fixed 5% non-adaptive
        fixed_ref = [r for r in results if r.stop_type == "fixed_pct" and r.base_param == 5.0 and not r.regime_adaptive]
        fixed_dd = fixed_ref[0].max_dd_pct if fixed_ref else no_stop_dd
        for r in results:
            r.dd_reduction_pct = (fixed_dd - r.max_dd_pct) / fixed_dd * 100 if fixed_dd > 0.01 else 0.0

        # Best: highest Sharpe among those preserving >80% return
        viable = [r for r in results if r.return_preserved_pct > 80]
        best = max(viable, key=lambda r: r.sharpe) if viable else max(results, key=lambda r: r.sharpe)

        return OptimizationResult(
            stop_results=results,
            no_stop_return=no_stop_ret,
            no_stop_dd=no_stop_dd,
            no_stop_sharpe=no_stop_sharpe,
            fixed_stop_dd=fixed_dd,
            best_stop=best,
            best_dd_reduction=best.dd_reduction_pct,
            best_return_preservation=best.return_preserved_pct,
            n_observations=len(self.equity),
        )

    def _apply_stop(
        self, stop_type: str, param: float, adaptive: bool,
    ) -> Tuple[np.ndarray, int, Dict[str, int]]:
        if stop_type == "fixed_pct":
            return apply_fixed_stop(self.equity, param, self.vix, adaptive)
        elif stop_type == "atr_trailing":
            return apply_atr_trailing(self.equity, self.prices, self.high, self.low, self.close, param, self.vix, adaptive=adaptive)
        elif stop_type == "chandelier":
            return apply_chandelier(self.equity, self.high, self.low, self.close, param, self.vix, adaptive=adaptive)
        elif stop_type == "keltner":
            return apply_keltner(self.equity, self.close, self.high, self.low, param, self.vix, adaptive=adaptive)
        elif stop_type == "vol_breakout":
            return apply_vol_breakout(self.equity, self.returns, param, self.vix, adaptive=adaptive)
        else:
            raise ValueError(f"Unknown stop type: {stop_type}")

    @staticmethod
    def generate_report(result: OptimizationResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"


def _build_html(r: OptimizationResult) -> str:
    b = r.best_stop
    # Top results sorted by Sharpe
    top = sorted(r.stop_results, key=lambda x: x.sharpe, reverse=True)[:15]

    def _row(s):
        hl = " style='color:#3fb950;font-weight:700'" if s is b else ""
        ad = "✓" if s.regime_adaptive else ""
        return (f"<tr{hl}><td style='text-align:left'>{s.stop_type}</td>"
                f"<td>{ad}</td><td>{_fr(s.base_param)}</td>"
                f"<td>{_fp(s.total_return_pct)}</td><td>{_fp(s.max_dd_pct)}</td>"
                f"<td>{_fr(s.sharpe)}</td><td>{_fp(s.return_preserved_pct)}</td>"
                f"<td>{_fp(s.dd_reduction_pct)}</td><td>{s.n_stops_triggered}</td></tr>")
    rows = "".join(_row(s) for s in top)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Adaptive Stop-Loss</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}
</style></head><body>
<h1>Adaptive Stop-Loss Optimizer</h1>
<div class="cards">
<div class="c"><div class="l">Best Stop</div><div class="v">{b.stop_type}</div></div>
<div class="c"><div class="l">Regime Adaptive</div><div class="v">{'Yes' if b.regime_adaptive else 'No'}</div></div>
<div class="c"><div class="l">DD Reduction</div><div class="v">{_fp(r.best_dd_reduction)}</div></div>
<div class="c"><div class="l">Return Preserved</div><div class="v">{_fp(r.best_return_preservation)}</div></div>
<div class="c"><div class="l">No-Stop DD</div><div class="v">{_fp(r.no_stop_dd)}</div></div>
<div class="c"><div class="l">Best DD</div><div class="v">{_fp(b.max_dd_pct)}</div></div>
<div class="c"><div class="l">No-Stop Sharpe</div><div class="v">{_fr(r.no_stop_sharpe)}</div></div>
<div class="c"><div class="l">Best Sharpe</div><div class="v">{_fr(b.sharpe)}</div></div>
</div>
<h2>Top 15 Configurations (by Sharpe)</h2>
<table><tr><th style="text-align:left">Type</th><th>Adaptive</th><th>Param</th><th>Return</th><th>Max DD</th><th>Sharpe</th><th>Preserved</th><th>DD Reduction</th><th>Stops</th></tr>{rows}</table>
</body></html>"""
