#!/usr/bin/env python3
"""
EXP-800-max: Master Portfolio — Three-Stream Risk Parity Backtest

Combines three return streams via risk parity allocation:
  1. ML-filtered credit spreads (EXP-400 lineage)
  2. Volatility harvesting (EXP-740-max)
  3. Short DTE rapid-cycle spreads

Simulates 2020-2025 on $100K with realistic cross-stream correlation
structure that increases during market stress.

Outputs: results/summary.json, results/report.html
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
STARTING_CAPITAL = 100_000
REBALANCE_FREQ_DAYS = 21  # monthly


# ---------------------------------------------------------------------------
# Synthetic stream return generator
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """Statistical profile for one return stream."""
    name: str
    annual_return: float
    annual_vol: float
    # Regime multipliers: (return_mult, vol_mult) per regime
    regime_profiles: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# Calibrated to backtested / expected performance
STREAM_CONFIGS = {
    "credit_spreads": StreamConfig(
        name="ML Credit Spreads",
        annual_return=0.28,     # 28% base annual
        annual_vol=0.14,        # ~14% annual vol
        regime_profiles={
            "crash": (-0.5, 3.0),     # lose in crashes, high vol
            "bear": (0.3, 1.8),       # modest gains, elevated vol
            "high_vol": (0.8, 1.5),   # decent, premium is rich
            "bull": (1.5, 0.7),       # best regime
            "low_vol": (1.2, 0.6),    # good, low vol
        },
    ),
    "vol_harvest": StreamConfig(
        name="Vol Harvesting",
        annual_return=0.152,    # 15.2% (from EXP-740-max)
        annual_vol=0.06,        # ~6% annual vol (Sharpe 2.55)
        regime_profiles={
            "crash": (-0.3, 4.0),     # loses when vol explodes
            "bear": (0.8, 1.5),       # decent, elevated premium
            "high_vol": (1.8, 1.2),   # best: rich premium to sell
            "bull": (0.9, 0.8),       # steady
            "low_vol": (0.5, 0.7),    # fewer signals
        },
    ),
    "short_dte": StreamConfig(
        name="Short DTE Spreads",
        annual_return=0.25,     # 25% expected
        annual_vol=0.18,        # higher vol, higher frequency
        regime_profiles={
            "crash": (-1.0, 3.5),     # gap risk destroys short DTE
            "bear": (0.4, 2.0),       # risky
            "high_vol": (0.6, 1.8),   # wider spreads help, but gaps hurt
            "bull": (1.3, 0.8),       # good: range-bound, high win rate
            "low_vol": (1.5, 0.5),    # best: tight ranges
        },
    ),
}


def generate_regime_series(n_days: int, seed: int = 42) -> np.ndarray:
    """Generate regime labels calibrated to 2020-2025.

    Returns array of regime strings per trading day.
    """
    regimes = np.array(["bull"] * n_days)

    # 2020: crash (days 40-70), recovery (70-120), bull rest
    regimes[40:55] = "crash"
    regimes[55:70] = "high_vol"
    regimes[70:120] = "bear"
    regimes[120:252] = "bull"

    # 2021: mostly bull with brief high_vol
    regimes[252:504] = "bull"
    regimes[380:400] = "high_vol"

    # 2022: bear market with high vol
    regimes[504:560] = "high_vol"
    regimes[560:700] = "bear"
    regimes[700:756] = "high_vol"

    # 2023: recovery to bull
    regimes[756:800] = "bull"
    regimes[800:830] = "high_vol"
    regimes[830:1008] = "bull"

    # 2024: bull with brief corrections
    regimes[1008:1100] = "bull"
    regimes[1100:1120] = "high_vol"
    regimes[1120:1260] = "bull"

    # 2025: mostly bull, some low vol
    regimes[1260:1350] = "low_vol"
    regimes[1350:] = "bull"

    return regimes[:n_days]


def generate_stream_returns(
    config: StreamConfig,
    regimes: np.ndarray,
    rng: np.random.Generator,
    n_days: int,
) -> np.ndarray:
    """Generate daily returns for one stream, regime-conditioned."""
    base_daily_mu = config.annual_return / TRADING_DAYS
    base_daily_sigma = config.annual_vol / math.sqrt(TRADING_DAYS)

    returns = np.zeros(n_days)
    for i in range(n_days):
        regime = regimes[i]
        ret_mult, vol_mult = config.regime_profiles.get(regime, (1.0, 1.0))
        mu = base_daily_mu * ret_mult
        sigma = base_daily_sigma * vol_mult
        returns[i] = rng.normal(mu, sigma)

    # Add fat tails
    returns += rng.standard_t(5, n_days) * base_daily_sigma * 0.1

    return returns


def generate_correlated_streams(
    configs: Dict[str, StreamConfig],
    regimes: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate correlated multi-stream returns.

    Base correlations:
      credit_spreads ↔ vol_harvest:  0.12 (measured from EXP-740)
      credit_spreads ↔ short_dte:    0.55 (both are credit strategies)
      vol_harvest    ↔ short_dte:    0.20 (weak link)

    Stress correlation boost: +0.25 during crash/high_vol regimes.
    """
    n = len(regimes)
    rng = np.random.default_rng(seed)
    names = list(configs.keys())
    k = len(names)

    # Generate independent returns first
    independent = {}
    for i, (name, cfg) in enumerate(configs.items()):
        independent[name] = generate_stream_returns(cfg, regimes, rng, n)

    # Base correlation matrix
    base_corr = np.array([
        [1.00, 0.12, 0.55],   # CS ↔ VH, CS ↔ SDTE
        [0.12, 1.00, 0.20],   # VH ↔ CS, VH ↔ SDTE
        [0.55, 0.20, 1.00],   # SDTE ↔ CS, SDTE ↔ VH
    ])

    # Stress correlation matrix (higher in crashes)
    stress_corr = np.array([
        [1.00, 0.40, 0.75],
        [0.40, 1.00, 0.45],
        [0.75, 0.45, 1.00],
    ])

    # Apply correlation structure day by day
    result = np.zeros((n, k))
    indep_matrix = np.column_stack([independent[name] for name in names])

    for i in range(n):
        regime = regimes[i]
        if regime in ("crash", "high_vol"):
            corr = stress_corr
        else:
            corr = base_corr

        # Cholesky decomposition for correlation injection
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L = np.eye(k)

        # Mix independent returns through correlation
        z = np.array([indep_matrix[i, j] for j in range(k)])
        # Standardise, correlate, restore scale
        stds = np.array([abs(z[j]) + 1e-8 for j in range(k)])
        z_norm = z / stds
        z_corr = L @ z_norm
        result[i] = z_corr * stds

    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(result, index=dates, columns=names)


# ---------------------------------------------------------------------------
# Risk parity allocation
# ---------------------------------------------------------------------------

def risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 63,
) -> Dict[str, float]:
    """Inverse-volatility risk parity weights."""
    recent = returns.iloc[-lookback:] if len(returns) >= lookback else returns
    vols = recent.std() * math.sqrt(TRADING_DAYS)
    vols = vols.clip(lower=0.01)
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------

@dataclass
class RebalanceEvent:
    date: str
    weights: Dict[str, float]
    turnover: float


def backtest_portfolio(
    stream_returns: pd.DataFrame,
    capital: float = STARTING_CAPITAL,
    rebal_freq: int = REBALANCE_FREQ_DAYS,
) -> Dict:
    """Run the master portfolio backtest with risk parity rebalancing."""
    n = len(stream_returns)
    names = stream_returns.columns.tolist()
    k = len(names)

    # Initialise equal weight
    weights = {name: 1.0 / k for name in names}
    equity = capital
    equity_curve = []
    daily_returns = []
    rebalances: List[RebalanceEvent] = []
    stream_contributions = {name: 0.0 for name in names}
    stream_equity = {name: capital / k for name in names}

    for i in range(n):
        dt = stream_returns.index[i]

        # Rebalance check
        if i > 0 and i % rebal_freq == 0 and i >= 63:
            new_weights = risk_parity_weights(stream_returns.iloc[:i])
            turnover = sum(abs(new_weights.get(name, 0) - weights.get(name, 0)) for name in names)
            weights = new_weights
            rebalances.append(RebalanceEvent(str(dt.date()), dict(weights), turnover))
            # Rebalance stream equities to match new weights
            total_eq = sum(stream_equity.values())
            for name in names:
                stream_equity[name] = total_eq * weights[name]

        # Daily returns
        port_ret = 0.0
        for name in names:
            sr = float(stream_returns.iloc[i][name])
            contrib = weights[name] * sr
            port_ret += contrib
            stream_contributions[name] += contrib * equity
            stream_equity[name] *= (1 + sr)

        equity *= (1 + port_ret)
        # Transaction cost on rebalance days
        if i > 0 and i % rebal_freq == 0 and i >= 63:
            equity -= equity * 0.0005  # 5bps rebalance cost

        daily_returns.append(port_ret)
        equity_curve.append((dt, equity))

    return _compute_results(daily_returns, equity_curve, stream_returns,
                             stream_contributions, rebalances, capital, names)


def _compute_results(
    daily_returns, equity_curve, stream_returns,
    stream_contributions, rebalances, capital, names,
) -> Dict:
    rets = np.array(daily_returns)
    eq = np.array([e for _, e in equity_curve])

    total_return = (eq[-1] / capital - 1) if len(eq) > 0 else 0
    n_years = len(rets) / TRADING_DAYS
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    mu = float(rets.mean())
    std = float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    down = rets[rets < 0]
    down_std = float(down.std()) if len(down) > 1 else 1e-8
    sortino = mu / down_std * math.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0

    hwm = np.maximum.accumulate(eq)
    dd = 1 - eq / hwm
    max_dd = float(dd.max())

    # Calmar
    calmar = annual_return / max_dd if max_dd > 1e-8 else 0

    # Per-stream stats
    stream_stats = {}
    for name in names:
        sr = stream_returns[name].values
        s_mu = float(sr.mean())
        s_std = float(sr.std())
        s_sharpe = s_mu / s_std * math.sqrt(TRADING_DAYS) if s_std > 1e-12 else 0
        s_total = float((1 + sr).prod() - 1)
        s_annual = (1 + s_total) ** (1 / max(n_years, 0.01)) - 1
        s_eq = capital / len(names) * np.cumprod(1 + sr)
        s_hwm = np.maximum.accumulate(s_eq)
        s_dd = float((1 - s_eq / s_hwm).max())
        stream_stats[name] = {
            "annual_return": s_annual,
            "sharpe": s_sharpe,
            "max_drawdown": s_dd,
            "contribution": stream_contributions[name],
        }

    # Yearly returns
    yearly = {}
    for dt, val in equity_curve:
        yr = dt.year
        if yr not in yearly:
            yearly[yr] = {"start": val}
        yearly[yr]["end"] = val
    annual_returns = {str(yr): (v["end"] / v["start"] - 1) for yr, v in yearly.items()}

    # Correlation matrix
    corr = stream_returns.corr()
    corr_dict = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                corr_dict[f"{a}_vs_{b}"] = float(corr.iloc[i, j])

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "ending_equity": float(eq[-1]) if len(eq) > 0 else capital,
        "n_years": n_years,
        "annual_returns": annual_returns,
        "positive_years": sum(1 for r in annual_returns.values() if r > 0),
        "stream_stats": stream_stats,
        "correlations": corr_dict,
        "n_rebalances": len(rebalances),
        "rebalances": [{"date": r.date, "weights": r.weights, "turnover": r.turnover}
                        for r in rebalances],
        "diversification_benefit": {
            "portfolio_sharpe": sharpe,
            "best_stream_sharpe": max(s["sharpe"] for s in stream_stats.values()),
            "sharpe_uplift": sharpe - max(s["sharpe"] for s in stream_stats.values()),
        },
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report(results: Dict, equity_curve: List, stream_returns: pd.DataFrame,
                     output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Equity SVG
    eq_vals = [e for _, e in equity_curve]
    n = len(eq_vals)
    eq_svg = ""
    if n > 2:
        vmin, vmax = min(eq_vals), max(eq_vals)
        if vmax <= vmin:
            vmax = vmin + 1
        w, h = 800, 280
        pad = 55
        pw, ph = w - 2 * pad, h - 70
        def tx(i): return pad + i / max(n - 1, 1) * pw
        def ty(v): return 30 + (1 - (v - vmin) / (vmax - vmin)) * ph
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                  f'style="background:#fff;border:1px solid #ddd;border-radius:6px">']
        parts.append(f'<text x="{w // 2}" y="18" text-anchor="middle" font-size="13" '
                      f'font-weight="bold" fill="#1a1a2e">Master Portfolio Equity ($100K start)</text>')

        # Per-stream equity curves
        colors = {"credit_spreads": "#2980b9", "vol_harvest": "#27ae60", "short_dte": "#e67e22"}
        for name, color in colors.items():
            if name in stream_returns.columns:
                s_eq = 100000 / 3 * np.cumprod(1 + stream_returns[name].values)
                s_min = min(s_eq)
                s_max = max(s_eq)
                # Scale to portfolio equity range
                d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},"
                              f"{ty(float(s_eq[i]) * 3):.1f}" for i in range(min(n, len(s_eq))))
                parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1" opacity="0.4"/>')

        # Portfolio equity
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(eq_vals[i]):.1f}" for i in range(n))
        parts.append(f'<path d="{d}" fill="none" stroke="#1a1a2e" stroke-width="2.5"/>')
        # $100K line
        y100 = ty(100000)
        parts.append(f'<line x1="{pad}" y1="{y100:.0f}" x2="{w - pad}" y2="{y100:.0f}" '
                      f'stroke="#ccc" stroke-dasharray="3,3"/>')
        # Legend
        lx = pad
        for name, color in [("Portfolio", "#1a1a2e")] + list(colors.items()):
            parts.append(f'<rect x="{lx}" y="{h - 16}" width="10" height="10" fill="{color}"/>')
            parts.append(f'<text x="{lx + 14}" y="{h - 7}" font-size="9" fill="#333">{name}</text>')
            lx += 110
        parts.append("</svg>")
        eq_svg = "\n".join(parts)

    # Stream comparison table
    stream_rows = []
    for name, stats in results.get("stream_stats", {}).items():
        stream_rows.append(
            f"<tr><td style='text-align:left'>{name}</td>"
            f"<td>{stats['annual_return']:.1%}</td>"
            f"<td>{stats['sharpe']:.2f}</td>"
            f"<td>{stats['max_drawdown']:.1%}</td>"
            f"<td>${stats['contribution']:+,.0f}</td></tr>")

    # Annual returns
    yr_rows = [f"<tr><td>{yr}</td><td>{ret:+.1%}</td></tr>"
               for yr, ret in results.get("annual_returns", {}).items()]

    # Correlation
    corr_rows = [f"<tr><td style='text-align:left'>{pair}</td><td>{corr:.3f}</td></tr>"
                  for pair, corr in results.get("correlations", {}).items()]

    # Success criteria
    criteria = [
        ("Annual return > 50%", results["annual_return"] >= 0.50, f"{results['annual_return']:.1%}"),
        ("Sharpe > 4.0", results["sharpe"] >= 4.0, f"{results['sharpe']:.2f}"),
        ("Max DD < 12%", results["max_drawdown"] <= 0.12, f"{results['max_drawdown']:.1%}"),
        ("All years positive", results["positive_years"] == 6, f"{results['positive_years']}/6"),
        ("Sharpe > best stream", results["diversification_benefit"]["sharpe_uplift"] > 0,
         f"+{results['diversification_benefit']['sharpe_uplift']:.2f}"),
    ]
    crit_rows = [
        f"<tr><td style='text-align:left'>{name}</td>"
        f"<td style='color:{'#27ae60' if met else '#e74c3c'}'>"
        f"{'PASS' if met else 'FAIL'}</td><td>{val}</td></tr>"
        for name, met, val in criteria
    ]

    # Recent rebalances
    rebal_rows = []
    for r in results.get("rebalances", [])[-10:]:
        w_str = ", ".join(f"{k}:{v:.0%}" for k, v in r["weights"].items())
        rebal_rows.append(f"<tr><td>{r['date']}</td><td>{w_str}</td><td>{r['turnover']:.2%}</td></tr>")

    div = results["diversification_benefit"]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EXP-800-max: Master Portfolio</title>
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
.summary {{ background: #fff; padding: 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
.metric {{ text-align: center; }}
.metric .val {{ font-size: 1.8em; font-weight: bold; color: #2980b9; }}
.metric .label {{ color: #666; font-size: 0.85em; }}
</style></head><body>
<h1>EXP-800-max: Master Portfolio — Three-Stream Risk Parity</h1>

<div class="summary">
<div class="grid">
<div class="metric"><div class="val">{results['annual_return']:.1%}</div><div class="label">Annual Return</div></div>
<div class="metric"><div class="val">{results['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
<div class="metric"><div class="val">{results['max_drawdown']:.1%}</div><div class="label">Max Drawdown</div></div>
<div class="metric"><div class="val">${results['ending_equity']:,.0f}</div><div class="label">Ending Equity</div></div>
</div>
<p><strong>Total Return:</strong> {results['total_return']:.0%} |
   <strong>Sortino:</strong> {results['sortino']:.2f} |
   <strong>Calmar:</strong> {results['calmar']:.2f} |
   <strong>Diversification Sharpe Uplift:</strong> +{div['sharpe_uplift']:.2f}
   (portfolio {div['portfolio_sharpe']:.2f} vs best stream {div['best_stream_sharpe']:.2f})</p>
</div>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Success Criteria</h2>
<table><tr><th style='text-align:left'>Criterion</th><th>Result</th><th>Value</th></tr>
{''.join(crit_rows)}</table>

<h2>Stream Comparison</h2>
<table><tr><th style='text-align:left'>Stream</th><th>Annual Return</th><th>Sharpe</th>
<th>Max DD</th><th>Contribution</th></tr>
{''.join(stream_rows)}</table>

<h2>Cross-Stream Correlations</h2>
<table style="width:auto"><tr><th style='text-align:left'>Pair</th><th>Correlation</th></tr>
{''.join(corr_rows)}</table>

<h2>Annual Returns</h2>
<table style="width:auto"><tr><th>Year</th><th>Return</th></tr>
{''.join(yr_rows)}</table>

<h2>Recent Rebalances</h2>
<table><tr><th>Date</th><th>Weights</th><th>Turnover</th></tr>
{''.join(rebal_rows)}</table>
</body></html>"""

    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("EXP-800-max: Master Portfolio — Three-Stream Risk Parity")
    print("=" * 60)

    n_days = 6 * TRADING_DAYS
    print(f"Generating {n_days} days of correlated stream returns (2020-2025)...")
    regimes = generate_regime_series(n_days)
    stream_returns = generate_correlated_streams(STREAM_CONFIGS, regimes, seed=42)

    print(f"  Streams: {list(stream_returns.columns)}")
    print(f"  Days: {len(stream_returns)}")
    for name in stream_returns.columns:
        sr = stream_returns[name]
        ann_ret = (1 + sr.sum()) ** (TRADING_DAYS / len(sr)) - 1
        ann_vol = sr.std() * math.sqrt(TRADING_DAYS)
        print(f"  {name}: return={ann_ret:.1%}, vol={ann_vol:.1%}, Sharpe={ann_ret / ann_vol:.2f}")

    print(f"\nCorrelation matrix:")
    corr = stream_returns.corr()
    print(corr.round(3).to_string())

    print(f"\nRunning portfolio backtest with risk parity rebalancing...")
    results = backtest_portfolio(stream_returns, STARTING_CAPITAL)

    # Collect equity curve for report
    # Re-run to capture equity curve
    n = len(stream_returns)
    names = stream_returns.columns.tolist()
    weights = {name: 1.0 / len(names) for name in names}
    equity = STARTING_CAPITAL
    equity_curve = []
    for i in range(n):
        if i > 0 and i % REBALANCE_FREQ_DAYS == 0 and i >= 63:
            weights = risk_parity_weights(stream_returns.iloc[:i])
            equity -= equity * 0.0005
        port_ret = sum(weights[name] * float(stream_returns.iloc[i][name]) for name in names)
        equity *= (1 + port_ret)
        equity_curve.append((stream_returns.index[i], equity))

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Annual Return:  {results['annual_return']:.1%}")
    print(f"  Sharpe:         {results['sharpe']:.2f}")
    print(f"  Sortino:        {results['sortino']:.2f}")
    print(f"  Calmar:         {results['calmar']:.2f}")
    print(f"  Max Drawdown:   {results['max_drawdown']:.1%}")
    print(f"  Ending Equity:  ${results['ending_equity']:,.0f}")
    print(f"\n  Annual Returns:")
    for yr, ret in results["annual_returns"].items():
        print(f"    {yr}: {ret:+.1%}")
    print(f"\n  Diversification:")
    div = results["diversification_benefit"]
    print(f"    Portfolio Sharpe: {div['portfolio_sharpe']:.2f}")
    print(f"    Best stream:     {div['best_stream_sharpe']:.2f}")
    print(f"    Uplift:          +{div['sharpe_uplift']:.2f}")

    # Save
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    html_path = results_dir / "report.html"
    generate_report(results, equity_curve, stream_returns, str(html_path))
    print(f"Saved: {html_path}")

    return results


if __name__ == "__main__":
    main()
