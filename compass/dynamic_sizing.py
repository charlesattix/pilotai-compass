"""
Dynamic position sizing framework for the Ultimate Portfolio.

Replaces static 1.6x leverage with signal-adaptive sizing:

Inputs:
  - VIX level (spot stress)
  - VIX term structure (VIX / VIX3M ratio)
  - 20-day realized vol
  - Regime classification (bull/bear/high_vol/low_vol/crisis/recovery)
  - Portfolio drawdown state

Output:
  - Leverage multiplier between 0.5x and 2.5x

Rules:
  - VIX > 30 → scale to 0.5x (crash protection)
  - VIX < 15 + uptrend → scale to 2.0-2.5x (capitalize on calm)
  - Drawdown circuit breaker: DD hits -8% → force 0.5x until recovery
  - Term structure inversion → reduce by 30%
  - Recovery after crisis → gradual ramp-up over 15 days

Backtest: compare static 1.6x vs dynamic on CAGR, DD, Sharpe, Calmar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DynamicSizingConfig:
    """All tuneable parameters."""
    # Leverage range
    max_leverage: float = 2.5
    min_leverage: float = 0.5
    default_leverage: float = 1.6

    # VIX-based scaling
    vix_boost_threshold: float = 15.0   # below → boost leverage
    vix_normal_low: float = 18.0
    vix_normal_high: float = 25.0
    vix_reduce_threshold: float = 30.0  # above → minimum leverage
    vix_crisis: float = 40.0

    # Boost in low-vol bull
    bull_boost_leverage: float = 2.3    # target in calm bull
    low_vol_max_leverage: float = 2.5   # VIX < 13, rock-bottom vol

    # VIX term structure
    ts_contango_boost: float = 0.90     # VIX/VIX3M < 0.90 → boost
    ts_inversion: float = 1.05          # VIX/VIX3M > 1.05 → reduce 30%
    ts_deep_inversion: float = 1.20     # deep inversion → minimum

    # Realized vol (annualized)
    rvol_low: float = 0.10              # boost allowed
    rvol_high: float = 0.25             # reduce
    rvol_extreme: float = 0.40          # minimum

    # Drawdown circuit breaker
    dd_trigger: float = 0.08            # -8% DD → force minimum
    dd_recovery: float = 0.03           # DD must recover to -3% to resume
    dd_min_leverage: float = 0.5        # leverage during circuit breaker

    # Trend detection (20-day SPY momentum)
    trend_bull_threshold: float = 0.02  # 2% 20d return = bullish
    trend_bear_threshold: float = -0.02

    # Smoothing
    smoothing_halflife: int = 3         # days (fast response)
    ramp_up_days: int = 15              # days to ramp back after crisis


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SizingState:
    """Daily sizing decision with full signal breakdown."""
    date: object
    leverage: float
    raw_leverage: float         # before smoothing
    vix: float
    vix_ratio: float
    realized_vol: float
    trend_20d: float
    drawdown: float
    regime: str
    circuit_breaker_active: bool
    # Signal breakdown
    vix_signal: float           # 0-1 (1 = max boost)
    ts_signal: float            # multiplier
    rvol_signal: float          # multiplier
    trend_signal: float         # multiplier


@dataclass
class BacktestComparison:
    """Static vs dynamic sizing comparison."""
    # Static
    static_leverage: float
    static_cagr: float
    static_sharpe: float
    static_dd: float
    static_calmar: float
    static_sortino: float
    static_vol: float
    static_equity: List[float]
    # Dynamic
    dynamic_cagr: float
    dynamic_sharpe: float
    dynamic_dd: float
    dynamic_calmar: float
    dynamic_sortino: float
    dynamic_vol: float
    dynamic_equity: List[float]
    dynamic_avg_leverage: float
    dynamic_max_leverage_used: float
    dynamic_min_leverage_used: float
    # Improvements
    cagr_improvement: float
    sharpe_improvement: float
    dd_improvement: float       # negative = better (less DD)
    calmar_improvement: float
    # Details
    states: List[SizingState]
    circuit_breaker_days: int
    n_days: int
    yearly_comparison: Dict[int, Dict[str, float]]


# ═══════════════════════════════════════════════════════════════════════════
# Core engine
# ═══════════════════════════════════════════════════════════════════════════


class DynamicSizer:
    """Computes time-varying portfolio leverage from market signals."""

    def __init__(self, config: Optional[DynamicSizingConfig] = None):
        self.cfg = config or DynamicSizingConfig()

    def compute_leverage(
        self,
        vix: float,
        vix_ratio: float,
        realized_vol: float,
        trend_20d: float,
        drawdown: float,
        circuit_breaker_active: bool,
    ) -> Tuple[float, str, Dict[str, float]]:
        """Compute target leverage from current market state.

        Returns: (leverage, regime, signal_breakdown)
        """
        cfg = self.cfg

        # ── Circuit breaker override ──────────────────────────────────────
        if circuit_breaker_active:
            return cfg.dd_min_leverage, "circuit_breaker", {
                "vix_signal": 0, "ts_signal": 1, "rvol_signal": 1, "trend_signal": 1}

        # ── 1. VIX signal → base leverage ─────────────────────────────────
        if vix <= cfg.vix_boost_threshold:
            # Low VIX = calm → boost
            vix_frac = 1.0 - (vix / cfg.vix_boost_threshold) * 0.3
            base = cfg.bull_boost_leverage + vix_frac * (cfg.low_vol_max_leverage - cfg.bull_boost_leverage)
            vix_signal = 1.0
        elif vix <= cfg.vix_normal_low:
            base = cfg.default_leverage + (cfg.bull_boost_leverage - cfg.default_leverage) * (
                1 - (vix - cfg.vix_boost_threshold) / (cfg.vix_normal_low - cfg.vix_boost_threshold))
            vix_signal = 0.7
        elif vix <= cfg.vix_normal_high:
            base = cfg.default_leverage
            vix_signal = 0.5
        elif vix <= cfg.vix_reduce_threshold:
            t = (vix - cfg.vix_normal_high) / (cfg.vix_reduce_threshold - cfg.vix_normal_high)
            base = cfg.default_leverage - t * (cfg.default_leverage - cfg.min_leverage * 1.5)
            vix_signal = 0.3
        else:
            base = cfg.min_leverage
            vix_signal = 0.0

        # ── 2. Term structure multiplier ──────────────────────────────────
        if vix_ratio < cfg.ts_contango_boost:
            ts_mult = 1.10  # healthy contango → slight boost
        elif vix_ratio < cfg.ts_inversion:
            ts_mult = 1.0   # normal
        elif vix_ratio < cfg.ts_deep_inversion:
            ts_mult = 0.70  # inversion → reduce 30%
        else:
            ts_mult = 0.50  # deep inversion → halve

        # ── 3. Realized vol multiplier ────────────────────────────────────
        if realized_vol < cfg.rvol_low:
            rvol_mult = 1.15  # low vol → slight boost
        elif realized_vol < cfg.rvol_high:
            t = (realized_vol - cfg.rvol_low) / (cfg.rvol_high - cfg.rvol_low)
            rvol_mult = 1.15 - t * 0.45  # ramp from 1.15 to 0.70
        elif realized_vol < cfg.rvol_extreme:
            rvol_mult = 0.60
        else:
            rvol_mult = 0.40

        # ── 4. Trend signal ───────────────────────────────────────────────
        if trend_20d > cfg.trend_bull_threshold:
            trend_mult = 1.10  # bullish trend → boost
        elif trend_20d < cfg.trend_bear_threshold:
            trend_mult = 0.80  # bearish → reduce
        else:
            trend_mult = 1.0

        # ── Combine ───────────────────────────────────────────────────────
        leverage = base * ts_mult * rvol_mult * trend_mult
        leverage = max(cfg.min_leverage, min(cfg.max_leverage, leverage))

        # ── Regime classification ─────────────────────────────────────────
        if vix > cfg.vix_crisis:
            regime = "crisis"
        elif vix > cfg.vix_reduce_threshold:
            regime = "high_vol"
        elif vix < cfg.vix_boost_threshold and trend_20d > 0:
            regime = "low_vol_bull"
        elif trend_20d > cfg.trend_bull_threshold:
            regime = "bull"
        elif trend_20d < cfg.trend_bear_threshold:
            regime = "bear"
        else:
            regime = "neutral"

        signals = {
            "vix_signal": round(vix_signal, 3),
            "ts_signal": round(ts_mult, 3),
            "rvol_signal": round(rvol_mult, 3),
            "trend_signal": round(trend_mult, 3),
        }
        return round(leverage, 4), regime, signals

    def backtest(
        self,
        portfolio_returns: np.ndarray,
        vix: np.ndarray,
        vix3m: np.ndarray,
        spy_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        starting_capital: float = 100_000,
    ) -> BacktestComparison:
        """Run static vs dynamic sizing comparison."""
        cfg = self.cfg
        n = len(portfolio_returns)

        # Rolling realized vol
        rvol = pd.Series(portfolio_returns).rolling(20, min_periods=5).std().fillna(0.012).values.copy()
        rvol *= math.sqrt(TRADING_DAYS)

        # 20-day trend
        spy_cum = np.cumsum(spy_returns)
        trend_20d = np.zeros(n)
        for i in range(20, n):
            trend_20d[i] = spy_cum[i] - spy_cum[i - 20]

        # VIX ratio
        vix_ratio = vix / np.maximum(vix3m, 1.0)

        # ── Static backtest ───────────────────────────────────────────────
        static_rets = portfolio_returns * cfg.default_leverage
        static_eq = _equity_curve(static_rets, starting_capital)
        static_m = _compute_metrics(static_rets)

        # ── Dynamic backtest ──────────────────────────────────────────────
        dynamic_rets = np.zeros(n)
        states: List[SizingState] = []
        prev_leverage = cfg.default_leverage
        cb_active = False  # circuit breaker
        ramp_counter = 0
        capital = starting_capital
        peak = capital

        for i in range(n):
            v = float(vix[i])
            vr = float(vix_ratio[i])
            rv = float(rvol[i])
            tr = float(trend_20d[i])
            dd = (peak - capital) / peak if peak > 0 else 0.0

            # Circuit breaker logic
            if dd >= cfg.dd_trigger:
                cb_active = True
                ramp_counter = 0
            elif cb_active and dd <= cfg.dd_recovery:
                cb_active = False
                ramp_counter = 0

            # Compute target leverage
            target, regime, signals = self.compute_leverage(
                v, vr, rv, tr, dd, cb_active)

            # Smooth (asymmetric: fast down, slow up)
            if cfg.smoothing_halflife > 0:
                alpha = 1 - math.exp(-math.log(2) / max(cfg.smoothing_halflife, 1))
                if target < prev_leverage:
                    eff_alpha = min(1.0, alpha * 3)  # fast down
                else:
                    eff_alpha = alpha * 0.5
                    if prev_leverage < cfg.default_leverage * 0.7:
                        ramp_counter += 1
                        ramp_t = min(1.0, ramp_counter / cfg.ramp_up_days)
                        eff_alpha *= ramp_t
                leverage = eff_alpha * target + (1 - eff_alpha) * prev_leverage
            else:
                leverage = target

            leverage = max(cfg.min_leverage, min(cfg.max_leverage, leverage))
            prev_leverage = leverage

            # Apply
            dynamic_rets[i] = portfolio_returns[i] * leverage
            capital *= (1 + dynamic_rets[i])
            capital = max(capital, 1.0)
            if capital > peak:
                peak = capital

            states.append(SizingState(
                date=dates[i], leverage=round(leverage, 4),
                raw_leverage=round(target, 4),
                vix=round(v, 1), vix_ratio=round(vr, 3),
                realized_vol=round(rv, 4), trend_20d=round(tr, 4),
                drawdown=round((peak - capital) / peak if peak > 0 else 0, 4),
                regime=regime, circuit_breaker_active=cb_active,
                vix_signal=signals["vix_signal"], ts_signal=signals["ts_signal"],
                rvol_signal=signals["rvol_signal"], trend_signal=signals["trend_signal"],
            ))

        dynamic_eq = _equity_curve(dynamic_rets, starting_capital)
        dynamic_m = _compute_metrics(dynamic_rets)

        # Yearly comparison
        yearly = _yearly_comparison(static_rets, dynamic_rets, dates)

        # Circuit breaker days
        cb_days = sum(1 for s in states if s.circuit_breaker_active)

        leverages = [s.leverage for s in states]

        return BacktestComparison(
            static_leverage=cfg.default_leverage,
            static_cagr=static_m["cagr_pct"],
            static_sharpe=static_m["sharpe"],
            static_dd=static_m["max_dd_pct"],
            static_calmar=static_m["calmar"],
            static_sortino=static_m["sortino"],
            static_vol=static_m["vol_pct"],
            static_equity=static_eq,
            dynamic_cagr=dynamic_m["cagr_pct"],
            dynamic_sharpe=dynamic_m["sharpe"],
            dynamic_dd=dynamic_m["max_dd_pct"],
            dynamic_calmar=dynamic_m["calmar"],
            dynamic_sortino=dynamic_m["sortino"],
            dynamic_vol=dynamic_m["vol_pct"],
            dynamic_equity=dynamic_eq,
            dynamic_avg_leverage=round(float(np.mean(leverages)), 3),
            dynamic_max_leverage_used=round(float(max(leverages)), 3),
            dynamic_min_leverage_used=round(float(min(leverages)), 3),
            cagr_improvement=round(dynamic_m["cagr_pct"] - static_m["cagr_pct"], 2),
            sharpe_improvement=round(dynamic_m["sharpe"] - static_m["sharpe"], 2),
            dd_improvement=round(dynamic_m["max_dd_pct"] - static_m["max_dd_pct"], 2),
            calmar_improvement=round(dynamic_m["calmar"] - static_m["calmar"], 2),
            states=states,
            circuit_breaker_days=cb_days,
            n_days=n,
            yearly_comparison=yearly,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _equity_curve(rets: np.ndarray, capital: float) -> List[float]:
    eq = [capital]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq


def _compute_metrics(rets: np.ndarray) -> dict:
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    ds = float(down.std()) if len(down) > 1 else std
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    return {
        "cagr_pct": round(cagr * 100, 2), "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2), "calmar": round(calmar, 2),
        "sortino": round(sortino, 2), "vol_pct": round(std * math.sqrt(TRADING_DAYS) * 100, 2),
    }


def _yearly_comparison(
    static: np.ndarray, dynamic: np.ndarray, dates: pd.DatetimeIndex,
) -> Dict[int, Dict[str, float]]:
    result = {}
    by_year: Dict[int, List[int]] = {}
    for i, d in enumerate(dates):
        by_year.setdefault(d.year, []).append(i)
    for yr, idx in sorted(by_year.items()):
        s_yr = static[idx]
        d_yr = dynamic[idx]
        s_m = _compute_metrics(s_yr)
        d_m = _compute_metrics(d_yr)
        result[yr] = {
            "static_cagr": s_m["cagr_pct"], "static_sharpe": s_m["sharpe"],
            "static_dd": s_m["max_dd_pct"],
            "dynamic_cagr": d_m["cagr_pct"], "dynamic_sharpe": d_m["sharpe"],
            "dynamic_dd": d_m["max_dd_pct"],
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Market data generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_market_data(n_years: float = 6.0, seed: int = 42) -> Dict[str, Any]:
    """Generate calibrated market data for backtest."""
    rng = np.random.RandomState(seed)
    n = int(n_years * TRADING_DAYS)
    idx = pd.bdate_range("2020-01-02", periods=n)

    # Portfolio returns (Ultimate Portfolio base, ~55% CAGR unlevered)
    port_mu = 0.55 / TRADING_DAYS
    port_sigma = 0.12 / math.sqrt(TRADING_DAYS)
    port_ret = rng.normal(port_mu, port_sigma, n)

    # SPY
    spy_ret = rng.normal(0.10 / TRADING_DAYS, 0.16 / math.sqrt(TRADING_DAYS), n)
    spy_ret = 0.3 * port_ret + 0.7 * spy_ret

    # VIX
    vix = np.zeros(n)
    vix[0] = 14.0
    for i in range(1, n):
        vix[i] = max(9, min(85, vix[i-1] + 0.03 * (16 - vix[i-1])
                             + rng.normal(0, 1.2) - spy_ret[i] * 150))

    # VIX3M
    vix3m = np.zeros(n)
    vix3m[0] = 16.0
    for i in range(1, n):
        vix3m[i] = max(10, min(60, vix3m[i-1] + 0.02 * (18 - vix3m[i-1])
                                + rng.normal(0, 0.8) - spy_ret[i] * 80))

    # Embed COVID (days 40-63)
    port_ret[40:63] = np.linspace(-0.04, -0.01, 23) * 1.5 + rng.normal(0, 0.005, 23)
    spy_ret[40:63] = np.linspace(-0.04, -0.01, 23) + rng.normal(0, 0.003, 23)
    vix[40:55] = np.linspace(15, 82, 15)
    vix[55:63] = np.linspace(82, 35, 8)
    vix3m[40:55] = np.linspace(16, 45, 15)
    vix3m[55:63] = np.linspace(45, 30, 8)

    # Embed 2022 bear (days 500-690)
    if n > 690:
        port_ret[500:690] = rng.normal(-0.001, port_sigma * 1.5, 190)
        spy_ret[500:690] = rng.normal(-0.0005, 0.012, 190)
        vix[500:690] = np.clip(25 + rng.normal(0, 3, 190), 18, 38)
        vix3m[500:690] = np.clip(22 + rng.normal(0, 2, 190), 16, 32)

    return {
        "portfolio_returns": port_ret,
        "spy_returns": spy_ret,
        "vix": vix,
        "vix3m": vix3m,
        "dates": idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════


def generate_report(
    result: BacktestComparison,
    output_path: str = "reports/dynamic_sizing_analysis.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Equity SVGs
    eq_svg = _build_dual_equity_svg(result.static_equity, result.dynamic_equity)
    lev_svg = _build_leverage_svg(result.states)

    # Comparison table
    def _delta(d, s, fmt=".1f", invert=False):
        diff = d - s
        if invert:
            diff = -diff
        c = "#16a34a" if diff > 0 else "#dc2626"
        return f'<span style="color:{c}">({diff:+{fmt}})</span>'

    yr_rows = ""
    for yr, d in sorted(result.yearly_comparison.items()):
        sc = "#16a34a" if d["dynamic_cagr"] > d["static_cagr"] else "#dc2626"
        yr_rows += f"""<tr><td>{yr}</td>
          <td>{d['static_cagr']:.1f}%</td><td>{d['static_sharpe']:.2f}</td><td>{d['static_dd']:.1f}%</td>
          <td style="color:{sc};font-weight:700">{d['dynamic_cagr']:.1f}%</td>
          <td>{d['dynamic_sharpe']:.2f}</td><td>{d['dynamic_dd']:.1f}%</td></tr>"""

    # Regime distribution
    regime_counts: Dict[str, int] = {}
    for s in result.states:
        regime_counts[s.regime] = regime_counts.get(s.regime, 0) + 1
    regime_rows = "".join(
        f"<tr><td>{r}</td><td>{c}</td><td>{c/result.n_days*100:.1f}%</td></tr>"
        for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]))

    winner = "Dynamic" if result.sharpe_improvement > 0 else "Static"
    wc = "#16a34a" if result.sharpe_improvement > 0 else "#dc2626"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Dynamic Sizing Analysis</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}td:first-child{{text-align:left}}
svg{{display:block;margin:0.5rem 0}}
.winner{{display:inline-block;padding:3px 12px;border-radius:4px;font-weight:700;font-size:0.82rem;background:{wc}15;color:{wc}}}
</style></head><body>
<h1>Dynamic Position Sizing Analysis</h1>
<p class="meta">Static 1.6x vs Adaptive 0.5x-2.5x | 2020-2025 | <span class="winner">{winner} Wins</span></p>

<div class="grid">
  <div class="card"><div class="l">Static Sharpe</div><div class="v">{result.static_sharpe:.2f}</div></div>
  <div class="card"><div class="l">Dynamic Sharpe</div><div class="v" style="color:{wc}">{result.dynamic_sharpe:.2f}</div></div>
  <div class="card"><div class="l">Delta Sharpe</div><div class="v" style="color:{'#16a34a' if result.sharpe_improvement > 0 else '#dc2626'}">{result.sharpe_improvement:+.2f}</div></div>
  <div class="card"><div class="l">Static CAGR</div><div class="v">{result.static_cagr:.1f}%</div></div>
  <div class="card"><div class="l">Dynamic CAGR</div><div class="v">{result.dynamic_cagr:.1f}%</div></div>
  <div class="card"><div class="l">Static DD</div><div class="v">{result.static_dd:.1f}%</div></div>
  <div class="card"><div class="l">Dynamic DD</div><div class="v" style="color:{'#16a34a' if result.dd_improvement < 0 else '#dc2626'}">{result.dynamic_dd:.1f}%</div></div>
  <div class="card"><div class="l">Avg Leverage</div><div class="v">{result.dynamic_avg_leverage:.2f}x</div></div>
  <div class="card"><div class="l">Leverage Range</div><div class="v">{result.dynamic_min_leverage_used:.1f}-{result.dynamic_max_leverage_used:.1f}x</div></div>
  <div class="card"><div class="l">CB Days</div><div class="v">{result.circuit_breaker_days}</div></div>
</div>

<h2>Head-to-Head Comparison</h2>
<table>
<tr><th>Metric</th><th>Static 1.6x</th><th>Dynamic</th><th>Delta</th></tr>
<tr><td>CAGR</td><td>{result.static_cagr:.1f}%</td><td>{result.dynamic_cagr:.1f}%</td><td style="color:{'#16a34a' if result.cagr_improvement > 0 else '#dc2626'};font-weight:700">{result.cagr_improvement:+.1f}%</td></tr>
<tr><td>Sharpe</td><td>{result.static_sharpe:.2f}</td><td>{result.dynamic_sharpe:.2f}</td><td style="color:{'#16a34a' if result.sharpe_improvement > 0 else '#dc2626'};font-weight:700">{result.sharpe_improvement:+.2f}</td></tr>
<tr><td>Max DD</td><td>{result.static_dd:.1f}%</td><td>{result.dynamic_dd:.1f}%</td><td style="color:{'#16a34a' if result.dd_improvement < 0 else '#dc2626'};font-weight:700">{result.dd_improvement:+.1f}%</td></tr>
<tr><td>Calmar</td><td>{result.static_calmar:.1f}</td><td>{result.dynamic_calmar:.1f}</td><td style="color:{'#16a34a' if result.calmar_improvement > 0 else '#dc2626'};font-weight:700">{result.calmar_improvement:+.1f}</td></tr>
<tr><td>Sortino</td><td>{result.static_sortino:.1f}</td><td>{result.dynamic_sortino:.1f}</td><td>{result.dynamic_sortino - result.static_sortino:+.1f}</td></tr>
<tr><td>Vol</td><td>{result.static_vol:.1f}%</td><td>{result.dynamic_vol:.1f}%</td><td>{result.dynamic_vol - result.static_vol:+.1f}%</td></tr>
</table>

<h2>Equity Curves</h2>
{eq_svg}

<h2>Dynamic Leverage Over Time</h2>
{lev_svg}

<h2>Yearly Breakdown</h2>
<table>
<tr><th>Year</th><th>S-CAGR</th><th>S-Sharpe</th><th>S-DD</th><th>D-CAGR</th><th>D-Sharpe</th><th>D-DD</th></tr>
{yr_rows}
</table>

<h2>Regime Distribution</h2>
<table><tr><th>Regime</th><th>Days</th><th>% of Total</th></tr>{regime_rows}</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/dynamic_sizing.py | VIX + TS + RVol + Trend + DD Circuit Breaker | Smoothed 3-day EMA</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


def _build_dual_equity_svg(static: List[float], dynamic: List[float]) -> str:
    if len(static) < 2:
        return ""
    w, h = 780, 220
    pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w - pl - pr, h - pt - pb
    all_vals = static + dynamic
    ym, yx = min(all_vals) * 0.95, max(all_vals) * 1.05
    n = len(static)

    def line(data, color):
        step = max(1, len(data) // 500)
        pts = [(i, data[i]) for i in range(0, len(data), step)]
        if pts[-1][0] != len(data) - 1:
            pts.append((len(data) - 1, data[-1]))
        def tx(i): return pl + i / max(n - 1, 1) * pw
        def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph
        return " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j, (i, v) in enumerate(pts))

    d1 = line(static, "#94a3b8")
    d2 = line(dynamic, "#16a34a")

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">Equity: Static (gray) vs Dynamic (green)</text>
  <path d="{d1}" fill="none" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <path d="{d2}" fill="none" stroke="#16a34a" stroke-width="1.5"/>
</svg>"""


def _build_leverage_svg(states: List[SizingState]) -> str:
    if not states:
        return ""
    w, h = 780, 150
    pl, pr, pt, pb = 65, 20, 22, 22
    pw, ph = w - pl - pr, h - pt - pb
    n = len(states)
    levs = [s.leverage for s in states]

    step = max(1, n // 500)
    pts = [(i, levs[i]) for i in range(0, n, step)]
    def tx(i): return pl + i / max(n - 1, 1) * pw
    def ty(v): return pt + (1 - (v - 0.0) / 3.0) * ph

    d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j, (i, v) in enumerate(pts))

    # 1.6x reference
    ref_y = ty(1.6)
    ref = f'<line x1="{pl}" y1="{ref_y:.0f}" x2="{w-pr}" y2="{ref_y:.0f}" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="4,3"/>'
    ref += f'<text x="{w-pr+2}" y="{ref_y:.0f}" font-size="9" fill="#94a3b8">1.6x static</text>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px">
  {ref}
  <path d="{d}" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="{w//2}" y="14" text-anchor="middle" font-size="11" fill="#64748b">Dynamic Leverage (0.5x-2.5x)</text>
</svg>"""


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def run_analysis(seed: int = 42) -> BacktestComparison:
    print("Dynamic Position Sizing Analysis")
    print("=" * 60)

    data = generate_market_data(seed=seed)
    sizer = DynamicSizer()
    result = sizer.backtest(
        data["portfolio_returns"], data["vix"], data["vix3m"],
        data["spy_returns"], data["dates"])

    print(f"\n  {'Metric':<15} {'Static 1.6x':>12} {'Dynamic':>12} {'Delta':>10}")
    print(f"  {'-'*49}")
    print(f"  {'CAGR':<15} {result.static_cagr:>11.1f}% {result.dynamic_cagr:>11.1f}% {result.cagr_improvement:>+9.1f}%")
    print(f"  {'Sharpe':<15} {result.static_sharpe:>12.2f} {result.dynamic_sharpe:>12.2f} {result.sharpe_improvement:>+10.2f}")
    print(f"  {'Max DD':<15} {result.static_dd:>11.1f}% {result.dynamic_dd:>11.1f}% {result.dd_improvement:>+9.1f}%")
    print(f"  {'Calmar':<15} {result.static_calmar:>12.1f} {result.dynamic_calmar:>12.1f} {result.calmar_improvement:>+10.1f}")
    print(f"  {'Sortino':<15} {result.static_sortino:>12.1f} {result.dynamic_sortino:>12.1f} {result.dynamic_sortino - result.static_sortino:>+10.1f}")
    print(f"\n  Avg leverage: {result.dynamic_avg_leverage:.2f}x "
          f"(range {result.dynamic_min_leverage_used:.1f}-{result.dynamic_max_leverage_used:.1f}x)")
    print(f"  Circuit breaker days: {result.circuit_breaker_days}")

    report = generate_report(result)
    print(f"\n  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
