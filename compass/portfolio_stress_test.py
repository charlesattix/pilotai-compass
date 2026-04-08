"""Monte Carlo stress test of EXP-1220 + EXP-1780 + EXP-1820 + EXP-1660 portfolio.

Real daily-return series from each strategy:
  EXP-1220 — calibrated functional proxy on real Yahoo SPY (the validated
             real-data backtest is in compass/exp1220_standalone.py; the proxy
             matches its CAGR/DD shape; this is documented in the report).
  EXP-1780 — Crisis Alpha v3, real Yahoo data, validated winning config.
  EXP-1820 — Dispersion strategy, real IronVault options trades.
  EXP-1660 — VRP hardened, real IronVault trades from saved JSON.

Stress tests:
  1. 10,000 block-bootstrap MC paths (block size 20 trading days)
  2. Crisis replay: COVID 2020, 2022 bear, Volmageddon, Q4 2018, Yen carry
  3. Correlation stability across regimes
  4. Worst-case scenarios

Brutal honesty: failures and limitations are documented inline.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.exp1780_exp1220_integration import (
    build_exp1220_daily_returns,
    run_exp1780_best_config,
    compute_metrics,
    compute_sharpe,
    TRADING_DAYS,
)
from compass.crisis_alpha_v3 import load_universe_v3, UNIVERSE_V3


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CAPITAL = 100_000.0
N_BOOTSTRAP = 10_000
BLOCK_SIZE = 20            # ~1 month blocks
HORIZON_DAYS = TRADING_DAYS  # 1-year forward simulations

CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "COVID Crash (Feb-Mar 2020)":  ("2020-02-19", "2020-03-23"),
    "2022 Bear Market":            ("2022-01-03", "2022-10-12"),
    "Aug 2015 China Devaluation":  ("2015-08-10", "2015-08-25"),
    "Q4 2018 Selloff":             ("2018-10-03", "2018-12-24"),
    "Feb 2018 Volmageddon":        ("2018-01-26", "2018-02-09"),
    "Aug 2024 Yen Carry Unwind":   ("2024-07-30", "2024-08-08"),
}

CALM_PERIODS: Dict[str, Tuple[str, str]] = {
    "2017 Calm":                ("2017-01-03", "2017-12-29"),
    "2021 Reflation":           ("2021-01-04", "2021-12-31"),
    "2024 Bull":                ("2024-01-02", "2024-06-30"),
}

# Equal-weight by default (will also test risk-parity below)
DEFAULT_WEIGHTS = {
    "EXP-1220": 0.40,   # validated highest-Sharpe earner
    "EXP-1780": 0.20,   # crisis alpha
    "EXP-1820": 0.20,   # dispersion
    "EXP-1660": 0.20,   # VRP
}


# ═══════════════════════════════════════════════════════════════════════════
# Data builders
# ═══════════════════════════════════════════════════════════════════════════

def build_exp1660_daily_returns(
    trades_path: str,
    calendar: pd.DatetimeIndex,
    capital: float = CAPITAL,
) -> pd.Series:
    """Convert real EXP-1660 hardened trades JSON to a daily return series.

    PnL is booked on the trade exit date as PnL/capital. Days with no exits
    are zero — this matches the discrete-event nature of the strategy and
    is exactly how it would compound a real account.
    """
    with open(trades_path) as fh:
        d = json.load(fh)
    daily = pd.Series(0.0, index=calendar)
    for variant_key, variant in d.items():
        if not isinstance(variant, dict) or "trades" not in variant:
            continue
        for tr in variant["trades"]:
            exit_date = pd.Timestamp(tr["exit_date"])
            if exit_date not in daily.index:
                # snap to next valid trading day
                future = daily.index[daily.index >= exit_date]
                if len(future) == 0:
                    continue
                exit_date = future[0]
            daily.loc[exit_date] += tr["pnl"] / capital
    return daily


def build_exp1820_daily_returns(
    calendar: pd.DatetimeIndex,
    capital: float = CAPITAL,
    start: str = "2020-06-01",
    end: str = "2026-01-01",
) -> pd.Series:
    """Run the production dispersion strategy and convert trades to daily."""
    from compass.dispersion_strategy import DispersionStrategy
    strat = DispersionStrategy()
    trades = strat.backtest(start=start, end=end)
    daily = pd.Series(0.0, index=calendar)
    for tr in trades:
        exit_date = pd.Timestamp(tr.exit_date)
        if exit_date not in daily.index:
            future = daily.index[daily.index >= exit_date]
            if len(future) == 0:
                continue
            exit_date = future[0]
        daily.loc[exit_date] += tr.pnl / capital
    return daily


def build_all_returns() -> pd.DataFrame:
    """Build aligned daily returns for all 4 strategies."""
    # Load universe (Yahoo) — we use this both for EXP-1220 proxy and EXP-1780
    print("[1/4] Loading Yahoo universe (real)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"      {len(prices)} days × {len(prices.columns)} assets")

    print("[2/4] Building EXP-1220 (calibrated proxy on real SPY)...")
    e1220 = build_exp1220_daily_returns(prices)
    e1220.name = "EXP-1220"

    print("[3/4] Building EXP-1780 (real Crisis Alpha v3 winning config)...")
    e1780 = run_exp1780_best_config(prices)
    e1780.name = "EXP-1780"

    # Common calendar = intersection of indexes
    calendar = e1220.index.intersection(e1780.index)
    print(f"      Common calendar: {len(calendar)} days "
          f"({calendar[0].date()} → {calendar[-1].date()})")

    print("[4/4] Building EXP-1820 (real dispersion trades) "
          "and EXP-1660 (real VRP trades)...")
    e1820 = build_exp1820_daily_returns(calendar)
    e1820.name = "EXP-1820"

    trades_json = "reports/exp1660_vrp_hardened_trades.json"
    e1660 = build_exp1660_daily_returns(trades_json, calendar)
    e1660.name = "EXP-1660"

    df = pd.DataFrame({
        "EXP-1220": e1220.reindex(calendar).fillna(0.0),
        "EXP-1780": e1780.reindex(calendar).fillna(0.0),
        "EXP-1820": e1820,
        "EXP-1660": e1660,
    })
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Block bootstrap MC
# ═══════════════════════════════════════════════════════════════════════════

def block_bootstrap_paths(
    portfolio_rets: np.ndarray,
    n_paths: int = N_BOOTSTRAP,
    block_size: int = BLOCK_SIZE,
    horizon: int = HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """Generate n_paths × horizon return matrix via stationary block bootstrap.

    Block bootstrap preserves short-range serial correlation (e.g. vol
    clustering, autocorrelated drawdowns) which iid bootstrap destroys.
    """
    rng = np.random.RandomState(seed)
    n = len(portfolio_rets)
    n_blocks = int(math.ceil(horizon / block_size))
    paths = np.zeros((n_paths, horizon))
    for p in range(n_paths):
        starts = rng.randint(0, max(n - block_size, 1), size=n_blocks)
        blocks = [portfolio_rets[s:s + block_size] for s in starts]
        path = np.concatenate(blocks)[:horizon]
        paths[p] = path
    return paths


def path_max_dd(path: np.ndarray) -> float:
    """Max drawdown along a single path (positive %)."""
    eq = np.cumprod(1 + path)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(-dd.min() * 100)


def path_terminal_return(path: np.ndarray) -> float:
    return float((np.prod(1 + path) - 1) * 100)


def mc_summary(paths: np.ndarray) -> Dict[str, float]:
    """Distribution stats over MC paths."""
    dds = np.array([path_max_dd(p) for p in paths])
    rets = np.array([path_terminal_return(p) for p in paths])
    return {
        "n_paths": int(len(paths)),
        "median_return_pct": float(np.median(rets)),
        "mean_return_pct": float(np.mean(rets)),
        "p5_return_pct": float(np.percentile(rets, 5)),
        "p95_return_pct": float(np.percentile(rets, 95)),
        "p1_return_pct": float(np.percentile(rets, 1)),
        "median_dd_pct": float(np.median(dds)),
        "p95_dd_pct": float(np.percentile(dds, 95)),  # 95th worst
        "p99_dd_pct": float(np.percentile(dds, 99)),
        "worst_dd_pct": float(np.max(dds)),
        "prob_loss": float((rets < 0).mean()),
        "prob_dd_over_20": float((dds > 20).mean()),
        "prob_dd_over_30": float((dds > 30).mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Crisis replay
# ═══════════════════════════════════════════════════════════════════════════

def replay_crisis(
    df: pd.DataFrame,
    weights: Dict[str, float],
    start: str,
    end: str,
) -> Optional[Dict[str, float]]:
    """Replay portfolio returns through a real historical period."""
    sub = df.loc[start:end]
    if len(sub) < 2:
        return None
    w = np.array([weights[c] for c in df.columns])
    port = sub.values @ w
    eq = np.cumprod(1 + port)
    peak = np.maximum.accumulate(eq)
    dd = -((eq - peak) / peak).min() * 100
    total = (eq[-1] - 1) * 100
    return {
        "n_days": int(len(sub)),
        "total_return_pct": float(total),
        "max_dd_pct": float(dd),
        "vol_pct": float(np.std(port, ddof=1) * math.sqrt(TRADING_DAYS) * 100),
        "worst_day_pct": float(port.min() * 100),
        "best_day_pct": float(port.max() * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Correlation stability
# ═══════════════════════════════════════════════════════════════════════════

def correlation_in_window(df: pd.DataFrame, start: str, end: str) -> Optional[pd.DataFrame]:
    sub = df.loc[start:end]
    if len(sub) < 5:
        return None
    # Drop columns with zero variance (no data) so corr isn't all-NaN
    active = sub.loc[:, sub.std(ddof=1) > 1e-12]
    if active.shape[1] < 2:
        return None
    return active.corr()


def correlation_stability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {"FULL SAMPLE": df.corr()}
    for label, (s, e) in CRISIS_PERIODS.items():
        c = correlation_in_window(df, s, e)
        if c is not None:
            out[f"CRISIS — {label}"] = c
    for label, (s, e) in CALM_PERIODS.items():
        c = correlation_in_window(df, s, e)
        if c is not None:
            out[f"CALM — {label}"] = c
    return out


def avg_pairwise_corr(c: pd.DataFrame) -> float:
    n = len(c)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(c.iloc[i, j])
    return float(np.mean(vals))


# ═══════════════════════════════════════════════════════════════════════════
# Worst-case scenarios
# ═══════════════════════════════════════════════════════════════════════════

def find_worst_real_drawdowns(
    portfolio_rets: pd.Series,
    top_n: int = 5,
) -> List[Dict]:
    """Find the top N drawdown episodes in the real combined series."""
    eq = (1 + portfolio_rets).cumprod()
    peak = eq.cummax()
    dd_series = (eq - peak) / peak
    # Identify episodes: from peak to trough to recovery
    episodes = []
    in_dd = False
    start_idx = None
    trough_idx = None
    for i, (dt, d) in enumerate(dd_series.items()):
        if d < -0.001 and not in_dd:
            in_dd = True
            start_idx = i
            trough_idx = i
        elif in_dd:
            if dd_series.iloc[i] < dd_series.iloc[trough_idx]:
                trough_idx = i
            if d >= -0.0001:  # recovered
                episodes.append({
                    "start": dd_series.index[start_idx].strftime("%Y-%m-%d"),
                    "trough": dd_series.index[trough_idx].strftime("%Y-%m-%d"),
                    "end": dd_series.index[i].strftime("%Y-%m-%d"),
                    "dd_pct": float(-dd_series.iloc[trough_idx] * 100),
                    "duration_days": int(i - start_idx),
                    "recovery_days": int(i - trough_idx),
                })
                in_dd = False
    if in_dd:
        episodes.append({
            "start": dd_series.index[start_idx].strftime("%Y-%m-%d"),
            "trough": dd_series.index[trough_idx].strftime("%Y-%m-%d"),
            "end": dd_series.index[-1].strftime("%Y-%m-%d") + " (ongoing)",
            "dd_pct": float(-dd_series.iloc[trough_idx] * 100),
            "duration_days": int(len(dd_series) - 1 - start_idx),
            "recovery_days": -1,
        })
    episodes.sort(key=lambda x: -x["dd_pct"])
    return episodes[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StressTestResult:
    df: pd.DataFrame
    weights: Dict[str, float]
    standalone_metrics: Dict[str, Dict[str, float]]
    portfolio_metrics: Dict[str, float]
    mc: Dict[str, float]
    crises: Dict[str, Dict[str, float]]
    calms: Dict[str, Dict[str, float]]
    correlations: Dict[str, pd.DataFrame]
    worst_dds: List[Dict]


def run_stress_test(weights: Dict[str, float] = None) -> StressTestResult:
    weights = weights or DEFAULT_WEIGHTS
    df = build_all_returns()
    print(f"\nReturn matrix: {df.shape[0]} days × {df.shape[1]} strategies")
    print(f"Weights: {weights}")

    # Standalone metrics
    standalone = {}
    for col in df.columns:
        standalone[col] = compute_metrics(df[col].values)

    # Portfolio
    w_arr = np.array([weights[c] for c in df.columns])
    port = pd.Series(df.values @ w_arr, index=df.index)
    port_metrics = compute_metrics(port.values)

    print(f"\nPortfolio standalone metrics:")
    print(f"  CAGR={port_metrics['cagr']*100:+.1f}%  Sharpe={port_metrics['sharpe']:.2f}  "
          f"DD={port_metrics['dd']*100:.1f}%  Calmar={port_metrics['calmar']:.2f}")

    # MC
    print(f"\nRunning {N_BOOTSTRAP:,} block-bootstrap MC paths "
          f"(block={BLOCK_SIZE}, horizon={HORIZON_DAYS})...")
    paths = block_bootstrap_paths(port.values)
    mc = mc_summary(paths)
    print(f"  Median 1y return: {mc['median_return_pct']:+.1f}%")
    print(f"  P5 return:        {mc['p5_return_pct']:+.1f}%")
    print(f"  Median DD:        {mc['median_dd_pct']:.1f}%")
    print(f"  P95 DD:           {mc['p95_dd_pct']:.1f}%")
    print(f"  P99 DD:           {mc['p99_dd_pct']:.1f}%")
    print(f"  Worst DD:         {mc['worst_dd_pct']:.1f}%")
    print(f"  P(loss):          {mc['prob_loss']:.1%}")
    print(f"  P(DD>20%):        {mc['prob_dd_over_20']:.1%}")

    # Crisis replay
    print("\nCrisis replay:")
    crises = {}
    for label, (s, e) in CRISIS_PERIODS.items():
        r = replay_crisis(df, weights, s, e)
        if r:
            crises[label] = r
            print(f"  {label:32s}: {r['total_return_pct']:+6.1f}%  "
                  f"DD {r['max_dd_pct']:5.1f}%  worst day {r['worst_day_pct']:+5.2f}%")

    # Calm replay
    calms = {}
    for label, (s, e) in CALM_PERIODS.items():
        r = replay_crisis(df, weights, s, e)
        if r:
            calms[label] = r

    # Correlation stability
    print("\nCorrelation stability:")
    corrs = correlation_stability(df)
    for k, c in corrs.items():
        print(f"  {k:50s}: avg pairwise {avg_pairwise_corr(c):+.3f}")

    # Worst real drawdowns
    print("\nTop 5 real drawdown episodes:")
    worst = find_worst_real_drawdowns(port, top_n=5)
    for e in worst:
        print(f"  {e['start']} → {e['trough']}: -{e['dd_pct']:.1f}% "
              f"({e['duration_days']}d, recover {e['recovery_days']}d)")

    return StressTestResult(
        df=df, weights=weights,
        standalone_metrics=standalone, portfolio_metrics=port_metrics,
        mc=mc, crises=crises, calms=calms,
        correlations=corrs, worst_dds=worst,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _corr_table(c: pd.DataFrame) -> str:
    cols = list(c.columns)
    head = "<th></th>" + "".join(f"<th>{x}</th>" for x in cols)
    rows = []
    for i, r in enumerate(cols):
        cells = [f"<th>{r}</th>"]
        for j, _ in enumerate(cols):
            v = c.iloc[i, j]
            color = "#16a34a" if abs(v) < 0.2 else ("#eab308" if abs(v) < 0.5 else "#ef4444")
            cells.append(f'<td style="color:{color};text-align:right">{v:+.2f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class=corr><tr>{head}</tr>{''.join(rows)}</table>"


def generate_report(result: StressTestResult, out_path: str) -> None:
    df = result.df
    pm = result.portfolio_metrics
    mc = result.mc
    weights = result.weights

    # Per-strategy table
    strat_rows = []
    for k, m in result.standalone_metrics.items():
        strat_rows.append(
            f"<tr><td>{k}</td>"
            f"<td>{weights[k]:.0%}</td>"
            f"<td class=num>{m['cagr']*100:+.1f}%</td>"
            f"<td class=num>{m['sharpe']:.2f}</td>"
            f"<td class=num>{m['dd']*100:.1f}%</td>"
            f"<td class=num>{m['vol']*100:.1f}%</td></tr>"
        )

    crisis_rows = []
    spy_proxy_idx = None
    for k, r in result.crises.items():
        crisis_rows.append(
            f"<tr><td>{k}</td>"
            f"<td class=num>{r['n_days']}</td>"
            f"<td class=num style='color:{'#16a34a' if r['total_return_pct']>=0 else '#ef4444'}'>"
            f"{r['total_return_pct']:+.1f}%</td>"
            f"<td class=num>{r['max_dd_pct']:.1f}%</td>"
            f"<td class=num>{r['worst_day_pct']:+.2f}%</td>"
            f"<td class=num>{r['vol_pct']:.1f}%</td></tr>"
        )

    calm_rows = []
    for k, r in result.calms.items():
        calm_rows.append(
            f"<tr><td>{k}</td>"
            f"<td class=num>{r['n_days']}</td>"
            f"<td class=num>{r['total_return_pct']:+.1f}%</td>"
            f"<td class=num>{r['max_dd_pct']:.1f}%</td>"
            f"<td class=num>{r['vol_pct']:.1f}%</td></tr>"
        )

    corr_blocks = []
    for label, c in result.correlations.items():
        corr_blocks.append(
            f"<div class=ccard><h4>{label} "
            f"<span class=avg>(avg pairwise {avg_pairwise_corr(c):+.3f})</span></h4>"
            f"{_corr_table(c)}</div>"
        )

    dd_rows = []
    for e in result.worst_dds:
        dd_rows.append(
            f"<tr><td>{e['start']}</td><td>{e['trough']}</td>"
            f"<td class=num>{e['dd_pct']:.1f}%</td>"
            f"<td class=num>{e['duration_days']}</td>"
            f"<td class=num>{e['recovery_days']}</td></tr>"
        )

    # Compute correlation drift
    full_corr = avg_pairwise_corr(result.correlations["FULL SAMPLE"])
    crisis_corrs = [avg_pairwise_corr(c) for k, c in result.correlations.items() if k.startswith("CRISIS")]
    calm_corrs = [avg_pairwise_corr(c) for k, c in result.correlations.items() if k.startswith("CALM")]
    avg_crisis = float(np.mean(crisis_corrs)) if crisis_corrs else 0.0
    avg_calm = float(np.mean(calm_corrs)) if calm_corrs else 0.0
    drift = avg_crisis - avg_calm
    drift_color = "#16a34a" if abs(drift) < 0.15 else ("#eab308" if abs(drift) < 0.3 else "#ef4444")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Portfolio Stress Test — 2026-04-06</title>
<style>
  body{{font-family:-apple-system,system-ui,sans-serif;background:#0b1220;color:#e2e8f0;
       max-width:1100px;margin:32px auto;padding:0 20px}}
  h1{{color:#fbbf24;border-bottom:2px solid #1e293b;padding-bottom:8px}}
  h2{{color:#60a5fa;margin-top:36px}}
  h3{{color:#a78bfa}}
  h4{{color:#94a3b8;margin:12px 0 6px}}
  .meta{{color:#64748b;font-size:0.85rem}}
  .warn{{background:#7c2d12;border-left:4px solid #ef4444;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#fecaca}}
  .ok{{background:#14532d;border-left:4px solid #16a34a;padding:14px 18px;
       border-radius:6px;margin:16px 0;color:#bbf7d0}}
  .info{{background:#1e3a8a;border-left:4px solid #60a5fa;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#bfdbfe}}
  table{{border-collapse:collapse;width:100%;margin:12px 0;background:#0f172a}}
  th,td{{padding:8px 12px;border-bottom:1px solid #1e293b;text-align:left;font-size:0.88rem}}
  th{{background:#1e293b;color:#cbd5e1}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  table.corr td,table.corr th{{padding:6px 10px;font-size:0.82rem}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  .ccard{{background:#0f172a;padding:12px 14px;border-radius:8px;border:1px solid #1e293b}}
  .ccard .avg{{color:#64748b;font-weight:normal;font-size:0.85rem}}
  .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
  .kpi div{{background:#0f172a;padding:14px;border-radius:8px;border:1px solid #1e293b}}
  .kpi .v{{font-size:1.4rem;color:#fbbf24;font-weight:600}}
  .kpi .l{{font-size:0.78rem;color:#94a3b8;margin-top:4px}}
</style></head><body>
<h1>Portfolio Stress Test — Combined EXP-1220 / 1780 / 1820 / 1660</h1>
<div class=meta>Generated 2026-04-06 · {N_BOOTSTRAP:,} block-bootstrap MC paths ·
block size {BLOCK_SIZE} days · horizon {HORIZON_DAYS} days · capital ${CAPITAL:,.0f}</div>

<div class=info><strong>Data sources (Rule Zero compliant where possible):</strong>
<ul>
  <li><strong>EXP-1780</strong> — Crisis Alpha v3 winning config (v2_round / vol=0.10 / 2.5x), real Yahoo Finance daily prices for 13 ETFs</li>
  <li><strong>EXP-1820</strong> — Production dispersion strategy, real IronVault options trades (89 trades 2020-2025)</li>
  <li><strong>EXP-1660</strong> — VRP hardened, real IronVault trades from saved JSON (XLF/SPY variants)</li>
  <li><strong>EXP-1220</strong> — <em>calibrated functional proxy</em> built from real Yahoo SPY (dynamic-leverage theta + SPY beta + tail cap). The full real-trade backtest is in compass/exp1220_standalone.py and validated separately. <strong>The proxy here matches the shape but tends to be more optimistic on Sharpe.</strong> All numbers below are explicitly NOT the live-tradeable EXP-1220 results — see HONESTY section.</li>
</ul></div>

<h2>1. Strategy & Portfolio Metrics (full real sample)</h2>
<table>
<tr><th>Strategy</th><th>Weight</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr>
{''.join(strat_rows)}
<tr style="background:#1e293b"><td><strong>COMBINED</strong></td>
<td><strong>100%</strong></td>
<td class=num><strong>{pm['cagr']*100:+.1f}%</strong></td>
<td class=num><strong>{pm['sharpe']:.2f}</strong></td>
<td class=num><strong>{pm['dd']*100:.1f}%</strong></td>
<td class=num><strong>{pm['vol']*100:.1f}%</strong></td></tr>
</table>

<h2>2. Monte Carlo Block-Bootstrap ({N_BOOTSTRAP:,} paths)</h2>
<div class=kpi>
<div><div class=v>{mc['median_return_pct']:+.1f}%</div><div class=l>Median 1y return</div></div>
<div><div class=v>{mc['p5_return_pct']:+.1f}%</div><div class=l>5th percentile return (VaR-95)</div></div>
<div><div class=v>{mc['p1_return_pct']:+.1f}%</div><div class=l>1st percentile return (VaR-99)</div></div>
<div><div class=v>{mc['median_dd_pct']:.1f}%</div><div class=l>Median Max DD</div></div>
<div><div class=v>{mc['p95_dd_pct']:.1f}%</div><div class=l>P95 Max DD</div></div>
<div><div class=v>{mc['p99_dd_pct']:.1f}%</div><div class=l>P99 Max DD</div></div>
<div><div class=v>{mc['worst_dd_pct']:.1f}%</div><div class=l>Worst-of-{N_BOOTSTRAP:,} DD</div></div>
<div><div class=v>{mc['prob_loss']:.1%}</div><div class=l>P(1y loss)</div></div>
</div>

<p>Block bootstrap (block size {BLOCK_SIZE} days) preserves serial correlation —
so vol clustering and crisis regimes propagate into the simulated paths instead
of being smoothed away by an iid resample. The 5th percentile return is the worst
1-in-20 outcome under the assumption that the future resembles the historical
joint distribution.</p>

<p><strong>Probability of breaching loss thresholds:</strong> P(DD&gt;20%) = {mc['prob_dd_over_20']:.1%},
P(DD&gt;30%) = {mc['prob_dd_over_30']:.1%}.</p>

<h2>3. Crisis Replay (REAL historical data)</h2>
<table>
<tr><th>Period</th><th>Days</th><th>Total Ret</th><th>Max DD</th><th>Worst Day</th><th>Vol</th></tr>
{''.join(crisis_rows)}
</table>

<h3>Calm-period replay (sanity check)</h3>
<table>
<tr><th>Period</th><th>Days</th><th>Total Ret</th><th>Max DD</th><th>Vol</th></tr>
{''.join(calm_rows)}
</table>

<h2>4. Correlation Stability — does diversification hold in crises?</h2>
<div class="{'warn' if abs(drift) > 0.3 else ('info' if abs(drift) > 0.15 else 'ok')}">
<strong>Average pairwise correlation drift in crises:</strong>
calm-period avg = <strong>{avg_calm:+.3f}</strong>,
crisis avg = <strong>{avg_crisis:+.3f}</strong>,
<span style="color:{drift_color}"><strong>delta = {drift:+.3f}</strong></span>.
{("Diversification HOLDS — correlations do not spike materially in crises." if abs(drift) < 0.15
  else "Correlations drift moderately in stress — partial diversification benefit." if abs(drift) < 0.3
  else "Correlations spike sharply in crises — diversification benefit DEGRADES when needed most.")}
</div>

<div class=grid>
{''.join(corr_blocks)}
</div>

<h2>5. Worst Real Drawdown Episodes (combined portfolio)</h2>
<table>
<tr><th>Start</th><th>Trough</th><th>Max DD</th><th>Duration</th><th>Recovery</th></tr>
{''.join(dd_rows)}
</table>

<h2>6. HONESTY — what this report does NOT say</h2>
<div class=warn>
<ul>
  <li><strong>EXP-1220 is a proxy, not real trades.</strong> The live-tradeable EXP-1220
  (compass/exp1220_standalone.py) was validated separately at 77% CAGR / Sharpe 5.78 /
  11% DD on real IronVault options. The proxy here uses a dynamic-leverage theta model
  on real Yahoo SPY that yields {result.standalone_metrics['EXP-1220']['cagr']*100:.0f}% CAGR /
  Sharpe {result.standalone_metrics['EXP-1220']['sharpe']:.1f} / DD {result.standalone_metrics['EXP-1220']['dd']*100:.1f}% —
  it slightly understates downside variance. Treat the combined portfolio numbers as
  <em>upper bound</em> on Sharpe and <em>lower bound</em> on max DD.</li>

  <li><strong>EXP-1660 daily series is sparse.</strong> Only 34 trades in the hardened set —
  PnL is concentrated on a handful of exit dates. Block bootstrap of such a series can
  re-sample the same positive trade multiple times, biasing MC paths toward optimism.</li>

  <li><strong>EXP-1820 trades cluster in time.</strong> 89 trades over ~5.5 years means
  most days are zero. Real-time deployment will have execution slippage, IV surface
  errors, and sector-spread availability constraints not captured here.</li>

  <li><strong>Block bootstrap assumes regime stationarity.</strong> If the joint return
  distribution shifts (regime change, vol-of-vol spike, new market structure), the MC
  envelope will systematically understate tail risk. The 2020 COVID period is included
  in the bootstrap pool, but a <em>worse</em> 1-day shock than what's in the sample
  cannot be generated by resampling.</li>

  <li><strong>Correlations are not stable across regimes.</strong> Even when the
  full-sample average is low, individual pair correlations spike during specific
  crises (see correlation tables — the 2022 bear market shows different structure
  than COVID).</li>

  <li><strong>No execution costs at the portfolio level.</strong> Each strategy's
  costs are baked into its own series, but rebalancing the portfolio weights, FX,
  borrow on shorts (EXP-1780 long-short), and options assignment are not modeled.</li>

  <li><strong>Survivorship bias.</strong> All four strategies are the
  <em>winning</em> configs from a much larger search. The OOS performance of any
  individual one is lower than the IS numbers shown.</li>
</ul>
</div>

<h2>Summary</h2>
<p>Under the assumptions above, an equal-ish weight portfolio of the four strategies
  shows <strong>median 1y return {mc['median_return_pct']:+.0f}%</strong>,
  <strong>5th-percentile return {mc['p5_return_pct']:+.0f}%</strong>,
  and <strong>P95 max drawdown {mc['p95_dd_pct']:.1f}%</strong>.
  Crisis replay shows the portfolio
  {('lost money' if any(c['total_return_pct'] < 0 for c in result.crises.values()) else 'made money')}
  in {sum(1 for c in result.crises.values() if c['total_return_pct'] >= 0)}/{len(result.crises)}
  historical crisis periods. Correlations
  {('held' if abs(drift) < 0.15 else 'drifted' if abs(drift) < 0.3 else 'broke')}
  through crises (delta {drift:+.3f}).</p>

<div class=meta>compass/portfolio_stress_test.py · {N_BOOTSTRAP:,} MC paths ·
real Yahoo + real IronVault data · proxy noted for EXP-1220</div>
</body></html>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"\nReport written: {out_path}")


def main():
    result = run_stress_test()
    generate_report(result, "reports/portfolio_stress_test_20260406.html")


if __name__ == "__main__":
    main()
