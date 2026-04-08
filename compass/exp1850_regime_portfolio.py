"""EXP-1850 — Regime-adaptive portfolio optimizer.

Combines three validated alpha streams under four allocation methods,
walk-forward validated on real data 2020-2025:

  Streams:
    - EXP-1220 dynamic (real Yahoo SPY+VIX+VIX3M, TailRiskProtector-scaled)
    - Crisis Alpha v5 hedge (real Yahoo 13-ETF, hedge-objective config)
    - VRP combined (4 ETFs SPY/QQQ/IWM/EEM, real Yahoo + FRED CBOE indices,
      walk-forward selected, vol-scaled to 5%/yr)

  Methods:
    1. STATIC — single max-Sharpe weight set fit on training window
    2. REGIME_CONDITIONAL — per-regime max-Sharpe; OOS picks today's regime
    3. RISK_PARITY_REGIME_TILT — 1/vol weights × regime favorability tilt
    4. MEAN_VARIANCE_REGIME_TILT — MVO per regime, tilted toward defensive
       allocations in BEAR/HIGH_VOL/CRASH regimes

  Walk-forward: 252-day train, 63-day test, step 63 days. Optimization is
  re-fit on every step using only data through the train cut-off.

  Reference benchmark: 2× EXP-1220 + 5% v5 hedge (the prior best combo).

Goal stated by Carlos: lift portfolio Sharpe from 3.88 → 6.0+.

Rule Zero: every input series traces to real Yahoo Finance, IronVault,
or FRED. No synthetic data, no random seeds, no in-fill imputation
beyond business-day forward-fill of options-cache gaps.

Output: compass/reports/exp1850_regime_portfolio.html
Cache:  compass/cache/exp1850_streams.pkl (force re-fetch with --no-cache)
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics, annualized_sharpe
from compass.regime import Regime, RegimeClassifier

CACHE_DIR = ROOT / "compass" / "cache"
CACHE_FILE = CACHE_DIR / "exp1850_streams.pkl"
REPORT_PATH = ROOT / "compass" / "reports" / "exp1850_regime_portfolio.html"
JSON_PATH = ROOT / "compass" / "reports" / "exp1850_regime_portfolio.json"

START = "2020-01-01"
END = "2025-12-31"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63
TARGET_VOL_VRP = 0.05  # 5%/yr vol target for VRP combined stream

STREAMS = ["exp1220", "v5_hedge", "vrp"]
REGIMES = [Regime.BULL.value, Regime.BEAR.value,
           Regime.HIGH_VOL.value, Regime.LOW_VOL.value, Regime.CRASH.value]


# ═══════════════════════════════════════════════════════════════════════════
# Stream loaders — REAL DATA ONLY
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220_stream() -> pd.Series:
    """Real Yahoo SPY + ^VIX + ^VIX3M, dynamic-leverage credit spread proxy."""
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    s = load_exp1220_dynamic()
    s.index = pd.DatetimeIndex(s.index)
    s.name = "exp1220"
    return s


def load_v5_hedge_stream(prices: pd.DataFrame) -> pd.Series:
    """Crisis Alpha v5 best hedge config (frozen from prior grid search).

    The 144-config grid in compass.crisis_alpha_v5 selected
        slow / vt=0.05 / l=1.0 / sg=0.05 / sh=2.0 / equity_short_only
    when scored by hedge_score against load_exp1220_dynamic. We hard-code
    that config here so we don't re-run the 2-min search every time.
    """
    from compass.crisis_alpha_v5 import HedgeConfigV5, backtest_v5

    cfg = HedgeConfigV5(
        name="v5_best_frozen",
        lookback_preset="slow",
        vol_target=0.05,
        leverage=1.0,
        dd_brake_threshold=0.05,
        dd_brake_zone=0.03,
        max_weight=0.20,
        require_confirmation=False,
        stress_threshold=0.05,
        stress_lookback=60,
        safe_haven_boost=2.0,
        equity_short_only=True,
    )
    r = backtest_v5(prices, cfg)
    s = r.daily_returns.copy()
    s.name = "v5_hedge"
    s.index = pd.DatetimeIndex(s.index)
    return s


def load_vrp_combined_stream() -> pd.Series:
    """VRP across SPY/QQQ/IWM/EEM, walk-forward, vol-targeted to 5%/yr."""
    from compass.exp1660_vrp_deepening import (
        compute_signals, load_pair, walk_forward, PAIRS,
    )

    daily = []
    for ticker in PAIRS:
        try:
            df = load_pair(ticker)
            sig = compute_signals(df)
            bt = walk_forward(ticker, sig)
        except Exception as e:
            print(f"  VRP {ticker}: SKIP ({e})")
            continue
        s = bt.daily_pnl.copy()
        s.index = pd.DatetimeIndex(s.index)
        daily.append(s.rename(ticker))

    if not daily:
        raise RuntimeError("VRP: no tickers loaded")

    df = pd.concat(daily, axis=1).fillna(0.0)
    combined = df.mean(axis=1)            # equal-weight 4 ETFs
    # Vol-target to 5% annual so the stream is comparable to the others
    realized_vol = combined.std() * math.sqrt(252)
    if realized_vol > 1e-9:
        combined = combined * (TARGET_VOL_VRP / realized_vol)
    combined.name = "vrp"
    return combined


def load_real_streams(use_cache: bool = True) -> Dict[str, pd.Series]:
    """Load (or fetch + cache) the three real-data return streams."""
    if use_cache and CACHE_FILE.exists():
        print(f"[cache] loading {CACHE_FILE.name}")
        with open(CACHE_FILE, "rb") as fh:
            return pickle.load(fh)

    from compass.crisis_alpha_v3 import load_universe_v3

    print("[load] real Yahoo universe (v3 13-ETF)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")

    print("[load] EXP-1220 dynamic stream...")
    exp1220 = load_exp1220_stream()
    print(f"       {len(exp1220)} days  CAGR "
          f"{full_metrics(exp1220.values)['cagr_pct']:+.1f}%")

    print("[load] Crisis Alpha v5 hedge stream...")
    v5 = load_v5_hedge_stream(prices)
    print(f"       {len(v5)} days  CAGR {full_metrics(v5.values)['cagr_pct']:+.1f}%")

    print("[load] VRP combined stream (SPY/QQQ/IWM/EEM)...")
    vrp = load_vrp_combined_stream()
    print(f"       {len(vrp)} days  CAGR {full_metrics(vrp.values)['cagr_pct']:+.1f}%")

    streams = {"exp1220": exp1220, "v5_hedge": v5, "vrp": vrp}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(streams, fh)
    print(f"[cache] saved → {CACHE_FILE}")
    return streams


def align_streams(streams: Dict[str, pd.Series]
                   ) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Align all streams to a common business-day index in [START, END]."""
    df = pd.concat([s.rename(k) for k, s in streams.items()], axis=1)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, df.index


# ═══════════════════════════════════════════════════════════════════════════
# Regime classification (real Yahoo SPY + ^VIX)
# ═══════════════════════════════════════════════════════════════════════════

def load_regime_series(index: pd.DatetimeIndex) -> pd.Series:
    """Tag each business day with a regime using real Yahoo SPY+^VIX."""
    from scripts.ultimate_portfolio import _fetch
    spy = _fetch("SPY", "2018-01-01", "2026-01-01")
    vix_df = _fetch("^VIX", "2018-01-01", "2026-01-01")
    vix = vix_df["Close"].squeeze()

    classifier = RegimeClassifier(trend_window=50, trend_threshold=5.0)
    series = classifier.classify_series(spy, vix)
    series = series.reindex(index, method="ffill")
    series = series.fillna(Regime.BULL).astype(str)
    return series


# ═══════════════════════════════════════════════════════════════════════════
# Allocation methods
# ═══════════════════════════════════════════════════════════════════════════

WEIGHT_CAP = 0.70    # no single stream above 70%
WEIGHT_FLOOR = 0.0   # long-only, allow zero
LEVERAGE_CAP = 2.0   # gross leverage cap on the post-allocation portfolio


def _ann_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 5 or rets.std(ddof=0) < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std(ddof=0) * math.sqrt(252))


def max_sharpe_weights(returns: pd.DataFrame,
                        weight_cap: float = WEIGHT_CAP) -> np.ndarray:
    """Long-only max-Sharpe weights via grid+scipy refinement.

    Falls back to equal-weight when sample is too small or covariance is
    degenerate.
    """
    n = returns.shape[1]
    if len(returns) < 30 or returns.std().min() < 1e-12:
        return np.ones(n) / n

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    # Try scipy SLSQP for max Sharpe
    try:
        from scipy.optimize import minimize

        def neg_sharpe(w):
            ret = float(np.dot(w, mu))
            vol = float(np.sqrt(np.dot(w, cov @ w)))
            if vol < 1e-9:
                return 1e9
            return -ret / vol

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(WEIGHT_FLOOR, weight_cap)] * n
        x0 = np.ones(n) / n
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 200})
        if res.success and np.isfinite(res.x).all():
            w = np.clip(res.x, WEIGHT_FLOOR, weight_cap)
            if w.sum() > 1e-9:
                return w / w.sum()
    except Exception:
        pass

    # Fallback: inverse-vol
    inv_vol = 1.0 / (returns.std().values + 1e-9)
    return inv_vol / inv_vol.sum()


def risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """Inverse-vol weights (true ERC simplified)."""
    vols = returns.std().values
    if (vols < 1e-12).any():
        n = returns.shape[1]
        return np.ones(n) / n
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


# Regime tilt: scale gross leverage of the portfolio by regime favorability.
# These tilts apply ON TOP of weight selection — they answer "how much
# total risk to take given today's regime?".
REGIME_LEVERAGE = {
    Regime.BULL.value:     1.80,   # press the trade
    Regime.LOW_VOL.value:  1.60,   # constructive but no momentum
    Regime.HIGH_VOL.value: 1.00,   # neutral
    Regime.BEAR.value:     0.70,   # lean defensive
    Regime.CRASH.value:    0.50,   # cut risk hard
}


def regime_conditional_weights(
    returns: pd.DataFrame, regime_train: pd.Series, weight_cap: float = WEIGHT_CAP,
) -> Dict[str, np.ndarray]:
    """Per-regime max-Sharpe weights computed on the training window.

    Falls back to global max-Sharpe weights when a regime has < 30 days.
    """
    out: Dict[str, np.ndarray] = {}
    global_w = max_sharpe_weights(returns, weight_cap=weight_cap)
    for r in REGIMES:
        mask = regime_train == r
        sub = returns.loc[mask]
        if len(sub) < 30:
            out[r] = global_w
        else:
            out[r] = max_sharpe_weights(sub, weight_cap=weight_cap)
    return out


def mean_variance_weights(
    returns: pd.DataFrame,
    weight_cap: float = WEIGHT_CAP,
    risk_aversion: float = 4.0,
) -> np.ndarray:
    """Long-only mean-variance optimization (max μ - λ·σ²)."""
    n = returns.shape[1]
    if len(returns) < 30:
        return np.ones(n) / n
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    try:
        from scipy.optimize import minimize

        def obj(w):
            return -(float(np.dot(w, mu)) - risk_aversion * float(np.dot(w, cov @ w)))

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(WEIGHT_FLOOR, weight_cap)] * n
        x0 = np.ones(n) / n
        res = minimize(obj, x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 200})
        if res.success:
            w = np.clip(res.x, WEIGHT_FLOOR, weight_cap)
            if w.sum() > 1e-9:
                return w / w.sum()
    except Exception:
        pass
    return max_sharpe_weights(returns, weight_cap=weight_cap)


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward portfolio runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WFResult:
    name: str
    daily_returns: pd.Series
    weights_history: pd.DataFrame      # rows = test step start, cols = streams
    leverage_history: pd.Series         # daily applied gross leverage
    metrics: Dict[str, float] = field(default_factory=dict)


def walk_forward_portfolio(
    returns: pd.DataFrame,
    regimes: pd.Series,
    method: str,
) -> WFResult:
    """Walk-forward backtest one allocation method."""
    n = len(returns)
    port_rets = pd.Series(0.0, index=returns.index)
    leverage_series = pd.Series(1.0, index=returns.index)
    weight_log: Dict[pd.Timestamp, np.ndarray] = {}

    start = TRAIN_DAYS
    while start + TEST_DAYS <= n:
        train = returns.iloc[start - TRAIN_DAYS:start]
        train_reg = regimes.iloc[start - TRAIN_DAYS:start]
        test = returns.iloc[start:start + TEST_DAYS]
        test_reg = regimes.iloc[start:start + TEST_DAYS]

        if method == "static":
            w = max_sharpe_weights(train)
            for i in range(len(test)):
                day = test.iloc[i].values
                lev = REGIME_LEVERAGE[Regime.HIGH_VOL.value]  # neutral
                lev = min(lev, LEVERAGE_CAP)
                port_rets.iloc[start + i] = float(np.dot(w, day)) * lev
                leverage_series.iloc[start + i] = lev
            weight_log[test.index[0]] = w

        elif method == "regime_conditional":
            wmap = regime_conditional_weights(train, train_reg)
            for i in range(len(test)):
                r = test_reg.iloc[i]
                w = wmap.get(r, wmap[Regime.BULL.value])
                lev = min(REGIME_LEVERAGE.get(r, 1.0), LEVERAGE_CAP)
                day = test.iloc[i].values
                port_rets.iloc[start + i] = float(np.dot(w, day)) * lev
                leverage_series.iloc[start + i] = lev
            weight_log[test.index[0]] = wmap[Regime.BULL.value]

        elif method == "risk_parity_regime_tilt":
            w = risk_parity_weights(train)
            for i in range(len(test)):
                r = test_reg.iloc[i]
                lev = min(REGIME_LEVERAGE.get(r, 1.0), LEVERAGE_CAP)
                day = test.iloc[i].values
                port_rets.iloc[start + i] = float(np.dot(w, day)) * lev
                leverage_series.iloc[start + i] = lev
            weight_log[test.index[0]] = w

        elif method == "mvo_regime_tilt":
            wmap_mv: Dict[str, np.ndarray] = {}
            global_mv = mean_variance_weights(train)
            for r in REGIMES:
                sub = train.loc[train_reg == r]
                if len(sub) < 30:
                    wmap_mv[r] = global_mv
                else:
                    # MVO is more risk-averse in defensive regimes
                    aversion = {
                        Regime.BULL.value: 2.0,
                        Regime.LOW_VOL.value: 3.0,
                        Regime.HIGH_VOL.value: 5.0,
                        Regime.BEAR.value: 8.0,
                        Regime.CRASH.value: 12.0,
                    }.get(r, 4.0)
                    wmap_mv[r] = mean_variance_weights(sub, risk_aversion=aversion)
            for i in range(len(test)):
                r = test_reg.iloc[i]
                w = wmap_mv.get(r, global_mv)
                lev = min(REGIME_LEVERAGE.get(r, 1.0), LEVERAGE_CAP)
                day = test.iloc[i].values
                port_rets.iloc[start + i] = float(np.dot(w, day)) * lev
                leverage_series.iloc[start + i] = lev
            weight_log[test.index[0]] = wmap_mv[Regime.BULL.value]

        else:
            raise ValueError(f"unknown method {method}")

        start += STEP_DAYS

    # Trim leading warmup zeros
    valid = port_rets.iloc[TRAIN_DAYS:start]
    valid_lev = leverage_series.iloc[TRAIN_DAYS:start]
    metrics = full_metrics(valid.values)

    weights_df = pd.DataFrame.from_dict(
        weight_log, orient="index", columns=returns.columns
    )

    return WFResult(
        name=method,
        daily_returns=valid,
        weights_history=weights_df,
        leverage_history=valid_lev,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reference benchmark — 2× EXP-1220 + 5% v5 hedge (the prior best combo)
# ═══════════════════════════════════════════════════════════════════════════

def reference_2x_5pct_hedge(returns: pd.DataFrame) -> WFResult:
    common = returns.index
    e = returns["exp1220"]
    h = returns["v5_hedge"]
    combined = (0.95) * e * 2.0 + 0.05 * h
    valid = combined.iloc[TRAIN_DAYS:]
    metrics = full_metrics(valid.values)
    return WFResult(
        name="benchmark_2x_5pct_v5",
        daily_returns=valid,
        weights_history=pd.DataFrame(),
        leverage_history=pd.Series(1.9, index=valid.index),
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Yearly + correlation reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def yearly_table(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().round(3)


def regime_distribution(regimes: pd.Series) -> Dict[str, float]:
    counts = regimes.value_counts(normalize=True)
    return {r: float(counts.get(r, 0.0)) for r in REGIMES}


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _metric_row(name: str, m: Dict, target_sharpe: float = 6.0) -> str:
    pass_color = "#16a34a" if m["sharpe"] >= target_sharpe else "#0f172a"
    return (
        f"<tr><td style='font-weight:700'>{name}</td>"
        f"<td>{m['cagr_pct']:.1f}%</td>"
        f"<td style='color:{pass_color};font-weight:700'>{m['sharpe']:.2f}</td>"
        f"<td>{m['max_dd_pct']:.1f}%</td>"
        f"<td>{m['calmar']:.2f}</td>"
        f"<td>{m['vol_pct']:.1f}%</td>"
        f"<td>{m['n_days']}</td></tr>"
    )


def _yearly_rows(yearly_by_method: Dict[str, List[Dict]]) -> Tuple[str, List[int]]:
    years = sorted({y["year"] for v in yearly_by_method.values() for y in v})
    rows = ""
    for yr in years:
        cells = ""
        for name in yearly_by_method.keys():
            row = next((y for y in yearly_by_method[name] if y["year"] == yr), {})
            cagr = row.get("cagr_pct", 0)
            sh = row.get("sharpe", 0)
            dd = row.get("max_dd_pct", 0)
            color = "#16a34a" if cagr > 0 else "#dc2626"
            cells += (
                f"<td style='color:{color}'>{cagr:.0f}%</td>"
                f"<td>{sh:.2f}</td><td>{dd:.1f}%</td>"
            )
        rows += f"<tr><td style='font-weight:700'>{yr}</td>{cells}</tr>"
    return rows, years


def _corr_rows(corr: pd.DataFrame) -> str:
    rows = ""
    for ix in corr.index:
        cells = ""
        for cx in corr.columns:
            v = corr.loc[ix, cx]
            color = "#16a34a" if v < 0 else ("#dc2626" if v > 0.5 else "#0f172a")
            cells += f"<td style='color:{color}'>{v:+.3f}</td>"
        rows += f"<tr><td style='font-weight:700'>{ix}</td>{cells}</tr>"
    return rows


def build_report(
    streams_df: pd.DataFrame,
    stream_metrics: Dict[str, Dict],
    method_results: Dict[str, WFResult],
    benchmark: WFResult,
    regimes: pd.Series,
    target_sharpe: float = 6.0,
) -> str:
    yearly_by_method = {
        name: yearly_table(r.daily_returns) for name, r in method_results.items()
    }
    yearly_by_method["benchmark_2x+5%v5"] = yearly_table(benchmark.daily_returns)

    method_rows = ""
    for name, r in method_results.items():
        method_rows += _metric_row(name, r.metrics, target_sharpe)
    method_rows += _metric_row("benchmark_2x+5%v5", benchmark.metrics, target_sharpe)

    stream_rows = ""
    for name in STREAMS:
        stream_rows += _metric_row(name, stream_metrics[name], target_sharpe)

    yearly_rows, years = _yearly_rows(yearly_by_method)
    corr = correlation_matrix(streams_df)
    corr_rows = _corr_rows(corr)
    regime_dist = regime_distribution(regimes)

    best_method = max(
        method_results.items(), key=lambda kv: kv[1].metrics["sharpe"]
    )
    best_name, best_r = best_method
    bench_sh = benchmark.metrics["sharpe"]
    sh_lift = best_r.metrics["sharpe"] - bench_sh
    target_gap = target_sharpe - best_r.metrics["sharpe"]
    target_color = "#16a34a" if target_gap <= 0 else "#dc2626"
    target_status = "REACHED" if target_gap <= 0 else f"GAP {target_gap:+.2f}"

    regime_dist_html = " · ".join(
        f"<strong>{k}</strong> {v*100:.0f}%" for k, v in regime_dist.items()
    )

    yearly_header_top = "".join(
        f"<th colspan='3'>{name}</th>" for name in yearly_by_method.keys()
    )
    yearly_header_bot = "".join(
        "<th>CAGR</th><th>Sharpe</th><th>DD</th>" for _ in yearly_by_method.keys()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1850 — Regime-Adaptive Portfolio Optimizer</title>
<style>
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b;line-height:1.5; }}
h1 {{ font-size:1.85em;color:#0f172a;margin-bottom:4px; }}
h2 {{ color:#334155;margin-top:2.4em;padding-bottom:8px;border-bottom:2px solid #e2e8f0; }}
.subtitle {{ color:#64748b;font-size:0.92rem;margin-bottom:24px; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px;margin:16px 0;font-size:0.84rem;line-height:1.7; }}
.verdict {{ background:#ecfdf5;border:2px solid {target_color};border-radius:10px;padding:20px;margin:24px 0; }}
.verdict h3 {{ margin-top:0;color:{target_color}; }}
.kpi-row {{ display:flex;gap:14px;flex-wrap:wrap;margin:18px 0; }}
.kpi {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;flex:1;min-width:130px; }}
.kpi .value {{ font-size:1.5em;font-weight:800;color:#0f172a; }}
.kpi .label {{ font-size:0.72em;color:#64748b;margin-top:4px;text-transform:uppercase; }}
table {{ width:100%;border-collapse:collapse;margin:14px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:10px 12px;text-align:right;font-weight:600;color:#475569;
     border-bottom:2px solid #cbd5e1;font-size:0.74em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:8px 12px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#f8fafc; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:12px 0; }}
.footer {{ margin-top:3em;padding-top:1em;border-top:1px solid #e2e8f0;font-size:0.78em;color:#94a3b8;text-align:center; }}
</style></head><body>

<h1>EXP-1850 — Regime-Adaptive Portfolio Optimizer</h1>
<div class="subtitle">3 streams × 4 allocation methods · walk-forward 2020–2025
on real Yahoo / FRED / IronVault data · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="sources">
<strong>Rule Zero — all data real:</strong><br>
<code>exp1220</code>: scripts.ultimate_portfolio.load_exp1220_dynamic
(real Yahoo SPY+^VIX+^VIX3M, TailRiskProtector-scaled dynamic leverage)<br>
<code>v5_hedge</code>: compass.crisis_alpha_v5 frozen best config
(slow / vt=0.05 / l=1.0 / sg=0.05 / sh=2.0 / equity-short-only) on real Yahoo 13-ETF universe<br>
<code>vrp</code>: compass.exp1660_vrp_deepening walk-forward on SPY/QQQ/IWM/EEM
(real Yahoo + FRED CBOE indices), equal-weight, vol-targeted to 5%/yr<br>
Regimes: compass.regime.RegimeClassifier on real Yahoo SPY+^VIX (50d MA, 5% trend threshold)<br>
Walk-forward: 252-day train · 63-day OOS test · quarterly step (re-fit each step)
</div>

<div class="verdict">
<h3>Best method: {best_name} → Sharpe {best_r.metrics['sharpe']:.2f} ({target_status} vs target {target_sharpe:.1f})</h3>
CAGR <strong>{best_r.metrics['cagr_pct']:.1f}%</strong> ·
Max DD <strong>{best_r.metrics['max_dd_pct']:.1f}%</strong> ·
Calmar <strong>{best_r.metrics['calmar']:.2f}</strong><br>
Sharpe lift vs prior best (2× EXP-1220 + 5% v5 hedge = {bench_sh:.2f}):
<strong>{sh_lift:+.2f}</strong>
</div>

<h2>1. Stream-level performance (full sample, no leverage)</h2>
<table>
<thead><tr><th>Stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>{stream_rows}</tbody>
</table>

<h2>2. Stream correlation matrix</h2>
<table>
<thead><tr><th></th>{''.join(f'<th>{c}</th>' for c in corr.columns)}</tr></thead>
<tbody>{corr_rows}</tbody>
</table>
<div class="note">
Negative entries (green) are the diversification engine. The lower the
off-diagonal, the larger the lift available from optimal allocation.
</div>

<h2>3. Walk-forward portfolio results (4 methods + benchmark)</h2>
<table>
<thead><tr><th>Method</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>OOS days</th></tr></thead>
<tbody>{method_rows}</tbody>
</table>
<div class="note">
Walk-forward setup: 252-day training window, 63-day out-of-sample test
window, quarterly re-fit. The benchmark (90% × 2× EXP-1220 + 5% v5 hedge)
uses no walk-forward — it is the static reference combo from prior work.
Sharpe lift &gt; 0 means the optimizer beats the static benchmark on the
identical data window.
</div>

<h2>4. Year-by-year comparison</h2>
<table>
<thead>
<tr><th rowspan='2'>Year</th>{yearly_header_top}</tr>
<tr>{yearly_header_bot}</tr>
</thead>
<tbody>{yearly_rows}</tbody>
</table>

<h2>5. Regime distribution (2020–2025 walk-forward sample)</h2>
<p>{regime_dist_html}</p>
<div class="note">
Regime classification uses VIX bands + SPY 50-day trend with shift-by-1
look-ahead protection. CRASH triggers on VIX&gt;40 with sharp decline.
</div>

<h2>6. Regime-conditional leverage profile</h2>
<table>
<thead><tr><th>Regime</th><th>Leverage</th><th>Rationale</th></tr></thead>
<tbody>
<tr><td>BULL</td><td>1.80×</td><td>Press the trade — credit spreads earn fastest in calm uptrends</td></tr>
<tr><td>LOW_VOL</td><td>1.60×</td><td>Constructive but no momentum kicker</td></tr>
<tr><td>HIGH_VOL</td><td>1.00×</td><td>Neutral — let the hedge breathe</td></tr>
<tr><td>BEAR</td><td>0.70×</td><td>Lean defensive, hedge starts contributing</td></tr>
<tr><td>CRASH</td><td>0.50×</td><td>Cut risk hard — survival mode</td></tr>
</tbody>
</table>

<h2>7. North Star gap analysis</h2>
<div class="kpi-row">
<div class="kpi"><div class="value">{best_r.metrics['cagr_pct']:.0f}%</div><div class="label">Best CAGR</div></div>
<div class="kpi"><div class="value">{best_r.metrics['sharpe']:.2f}</div><div class="label">Best Sharpe</div></div>
<div class="kpi"><div class="value">{best_r.metrics['max_dd_pct']:.1f}%</div><div class="label">Best Max DD</div></div>
<div class="kpi"><div class="value">6.00</div><div class="label">North Star Sharpe</div></div>
<div class="kpi"><div class="value" style="color:{target_color}">{target_status}</div><div class="label">Status</div></div>
</div>

<div class="footer">
compass/exp1850_regime_portfolio.py · 4 methods × walk-forward · Rule Zero
(real Yahoo + FRED + IronVault, no synthetic data)
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true", help="force re-fetch all streams")
    args = p.parse_args()

    print("=" * 72)
    print("EXP-1850 — Regime-Adaptive Portfolio Optimizer")
    print("=" * 72)

    streams = load_real_streams(use_cache=not args.no_cache)
    aligned, idx = align_streams(streams)
    print(f"\n[align] {len(aligned)} business days, "
          f"{aligned.index.min().date()} → {aligned.index.max().date()}")

    print("[regime] classifying days from real SPY+^VIX...")
    regimes = load_regime_series(idx)
    dist = regimes.value_counts(normalize=True).to_dict()
    print(f"[regime] {dict((k, round(v, 3)) for k, v in dist.items())}")

    stream_metrics = {k: full_metrics(aligned[k].values) for k in STREAMS}
    print("\n[streams] standalone metrics (no leverage):")
    for k in STREAMS:
        m = stream_metrics[k]
        print(f"  {k:12s}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:5.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Vol {m['vol_pct']:5.1f}%")

    print("\n[corr] stream correlation matrix")
    print(correlation_matrix(aligned).to_string())

    print("\n[walk-forward] running 4 allocation methods...")
    methods = ["static", "regime_conditional", "risk_parity_regime_tilt", "mvo_regime_tilt"]
    results: Dict[str, WFResult] = {}
    for m in methods:
        r = walk_forward_portfolio(aligned, regimes, m)
        results[m] = r
        print(f"  {m:30s}  CAGR {r.metrics['cagr_pct']:+7.1f}%  "
              f"Sharpe {r.metrics['sharpe']:5.2f}  DD {r.metrics['max_dd_pct']:5.1f}%  "
              f"Calmar {r.metrics['calmar']:5.2f}")

    print("\n[benchmark] 2× EXP-1220 + 5% v5 hedge (prior best combo)")
    bench = reference_2x_5pct_hedge(aligned)
    print(f"  benchmark_2x+5%v5             CAGR {bench.metrics['cagr_pct']:+7.1f}%  "
          f"Sharpe {bench.metrics['sharpe']:5.2f}  DD {bench.metrics['max_dd_pct']:5.1f}%")

    best_name = max(results.items(), key=lambda kv: kv[1].metrics["sharpe"])[0]
    best = results[best_name]
    print(f"\n[best] {best_name}: Sharpe {best.metrics['sharpe']:.2f} "
          f"(target 6.00, gap {6.0 - best.metrics['sharpe']:+.2f}) "
          f"vs benchmark {bench.metrics['sharpe']:.2f} (lift {best.metrics['sharpe'] - bench.metrics['sharpe']:+.2f})")

    print("\n[report] generating HTML + JSON...")
    html = build_report(aligned, stream_metrics, results, bench, regimes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}  ({len(html)/1024:.0f} KB)")

    import json
    summary = {
        "experiment": "EXP-1850",
        "title": "Regime-Adaptive Portfolio Optimizer",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_window": {
            "start": str(aligned.index.min().date()),
            "end": str(aligned.index.max().date()),
            "n_days": int(len(aligned)),
        },
        "rule_zero": {
            "synthetic_data": False,
            "sources": {
                "exp1220": "scripts.ultimate_portfolio.load_exp1220_dynamic (Yahoo SPY+^VIX+^VIX3M)",
                "v5_hedge": "compass.crisis_alpha_v5 frozen best (Yahoo 13-ETF)",
                "vrp": "compass.exp1660_vrp_deepening WF (Yahoo + FRED CBOE indices, 4 ETFs)",
                "regimes": "compass.regime.RegimeClassifier on Yahoo SPY+^VIX",
            },
        },
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "leverage_cap": LEVERAGE_CAP,
            "weight_cap": WEIGHT_CAP,
        },
        "stream_metrics": {k: stream_metrics[k] for k in STREAMS},
        "stream_correlation": correlation_matrix(aligned).to_dict(),
        "regime_distribution": regime_distribution(regimes),
        "regime_leverage": REGIME_LEVERAGE,
        "methods": {
            name: {
                "metrics": r.metrics,
                "yearly": yearly_table(r.daily_returns),
                "n_oos_days": int(len(r.daily_returns)),
            }
            for name, r in results.items()
        },
        "benchmark": {
            "name": "2x_exp1220 + 5%_v5_hedge",
            "metrics": bench.metrics,
            "yearly": yearly_table(bench.daily_returns),
        },
        "best_method": best_name,
        "north_star": {
            "target_sharpe": 6.0,
            "best_sharpe": best.metrics["sharpe"],
            "gap": round(6.0 - best.metrics["sharpe"], 3),
            "achieved": best.metrics["sharpe"] >= 6.0,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  → {JSON_PATH}")


if __name__ == "__main__":
    main()
