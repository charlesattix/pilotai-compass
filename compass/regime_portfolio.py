"""
Regime-adaptive portfolio allocator for the Ultimate Portfolio.

Shifts strategy weights based on regime detector output:
  BULL    → EXP-1220 at 2.0x, reduce hedges to 2%
  BEAR    → EXP-1220 at 0.8x, pairs 15%, vol term 10%
  CRASH   → all positions cut 50%, activate tail hedges
  HIGH_VOL→ vol term structure 15%, EXP-1220 at 1.0x
  LOW_VOL → EXP-1220 at 2.0x, reduce diversifiers

Backtests 2020-2025: static weights vs regime-adaptive vs dynamic sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Strategy IDs (same as production_portfolio_wf.py)
STRATEGY_IDS = [
    "EXP-1220_DynLev", "CrossAsset_Pairs", "VolTermStructure",
    "TLT_IronCondors", "XLI_IronCondors",
]

SHORT_NAMES = {
    "EXP-1220_DynLev": "EXP-1220",
    "CrossAsset_Pairs": "Pairs",
    "VolTermStructure": "VolTerm",
    "TLT_IronCondors": "TLT IC",
    "XLI_IronCondors": "XLI IC",
}

# Strategy return profiles (calibrated to IronVault backtests)
STRATEGY_PROFILES = {
    "EXP-1220_DynLev": {"annual_return": 0.77, "annual_vol": 0.14, "crisis_beta": 0.8},
    "CrossAsset_Pairs": {"annual_return": 0.15, "annual_vol": 0.06, "crisis_beta": -0.10},
    "VolTermStructure": {"annual_return": 0.12, "annual_vol": 0.08, "crisis_beta": 0.30},
    "TLT_IronCondors": {"annual_return": 0.18, "annual_vol": 0.10, "crisis_beta": 0.40},
    "XLI_IronCondors": {"annual_return": 0.20, "annual_vol": 0.11, "crisis_beta": 0.60},
}

CORRELATIONS = {
    ("EXP-1220_DynLev", "CrossAsset_Pairs"): 0.05,
    ("EXP-1220_DynLev", "VolTermStructure"): 0.25,
    ("EXP-1220_DynLev", "TLT_IronCondors"): -0.10,
    ("EXP-1220_DynLev", "XLI_IronCondors"): 0.30,
    ("CrossAsset_Pairs", "VolTermStructure"): 0.10,
    ("CrossAsset_Pairs", "TLT_IronCondors"): -0.05,
    ("CrossAsset_Pairs", "XLI_IronCondors"): 0.08,
    ("VolTermStructure", "TLT_IronCondors"): 0.15,
    ("VolTermStructure", "XLI_IronCondors"): 0.20,
    ("TLT_IronCondors", "XLI_IronCondors"): -0.12,
}


# ═══════════════════════════════════════════════════════════════════════════
# Allocation tables
# ═══════════════════════════════════════════════════════════════════════════

# Static baseline: weight_hint-proportional allocation
STATIC_WEIGHTS = {
    "EXP-1220_DynLev": 0.55,
    "CrossAsset_Pairs": 0.15,
    "VolTermStructure": 0.10,
    "TLT_IronCondors": 0.10,
    "XLI_IronCondors": 0.10,
}
STATIC_LEVERAGE = 1.6

# Regime-adaptive allocation + leverage
REGIME_ALLOCATION = {
    "bull": {
        "leverage": 2.0,
        "weights": {
            "EXP-1220_DynLev": 0.65,
            "CrossAsset_Pairs": 0.10,
            "VolTermStructure": 0.05,
            "TLT_IronCondors": 0.10,
            "XLI_IronCondors": 0.10,
        },
    },
    "bear": {
        "leverage": 0.8,
        "weights": {
            "EXP-1220_DynLev": 0.40,
            "CrossAsset_Pairs": 0.25,
            "VolTermStructure": 0.15,
            "TLT_IronCondors": 0.10,
            "XLI_IronCondors": 0.10,
        },
    },
    "crash": {
        "leverage": 0.5,  # all positions cut 50% via leverage
        "weights": {
            "EXP-1220_DynLev": 0.20,
            "CrossAsset_Pairs": 0.30,  # crisis diversifier
            "VolTermStructure": 0.20,
            "TLT_IronCondors": 0.15,  # flight to quality
            "XLI_IronCondors": 0.15,
        },
    },
    "high_vol": {
        "leverage": 1.0,
        "weights": {
            "EXP-1220_DynLev": 0.35,
            "CrossAsset_Pairs": 0.20,
            "VolTermStructure": 0.20,
            "TLT_IronCondors": 0.15,
            "XLI_IronCondors": 0.10,
        },
    },
    "low_vol": {
        "leverage": 2.0,
        "weights": {
            "EXP-1220_DynLev": 0.70,
            "CrossAsset_Pairs": 0.08,
            "VolTermStructure": 0.07,
            "TLT_IronCondors": 0.08,
            "XLI_IronCondors": 0.07,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DayState:
    date: object
    regime: str
    leverage: float
    weights: Dict[str, float]
    method: str  # "static", "regime_adaptive", "dynamic_sizing"


@dataclass
class MethodResult:
    name: str
    cagr: float
    sharpe: float
    max_dd: float
    calmar: float
    sortino: float
    vol: float
    equity: List[float]
    daily_returns: np.ndarray
    yearly: Dict[int, Dict[str, float]]
    avg_leverage: float
    states: List[DayState]


@dataclass
class ComparisonResult:
    methods: Dict[str, MethodResult]
    regime_distribution: Dict[str, int]
    n_days: int
    dates: Any  # pd.DatetimeIndex
    winner_sharpe: str
    winner_calmar: str


# ═══════════════════════════════════════════════════════════════════════════
# Return stream generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_data(n_years: float = 6.0, seed: int = 42) -> Dict[str, Any]:
    """Generate correlated strategy returns + market data."""
    rng = np.random.RandomState(seed)
    n = int(n_years * TRADING_DAYS)
    idx = pd.bdate_range("2020-01-02", periods=n)

    # Correlated normals
    ns = len(STRATEGY_IDS)
    corr = np.eye(ns)
    for i, si in enumerate(STRATEGY_IDS):
        for j, sj in enumerate(STRATEGY_IDS):
            if i == j:
                continue
            key = (si, sj) if (si, sj) in CORRELATIONS else (sj, si)
            if key in CORRELATIONS:
                corr[i, j] = CORRELATIONS[key]
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    Z = rng.randn(n, ns) @ L.T

    strat_returns = {}
    for i, sid in enumerate(STRATEGY_IDS):
        p = STRATEGY_PROFILES[sid]
        mu = p["annual_return"] / TRADING_DAYS
        sigma = p["annual_vol"] / math.sqrt(TRADING_DAYS)
        rets = mu + sigma * Z[:, i]
        # COVID
        cb = p["crisis_beta"]
        rets[40:63] = np.linspace(-0.04, -0.01, 23) * cb + rng.normal(0, 0.005, 23)
        # 2022 bear
        if n > 690:
            bear_daily = -0.15 / 190 * cb
            rets[500:690] = rng.normal(bear_daily, abs(bear_daily) * 0.8, 190)
        strat_returns[sid] = rets

    # VIX
    spy_ret = rng.normal(0.0004, 0.01, n)
    vix = np.zeros(n); vix[0] = 14.0
    for i in range(1, n):
        vix[i] = max(9, min(85, vix[i-1] + 0.03*(16-vix[i-1]) + rng.normal(0, 1.2) - spy_ret[i]*150))
    vix[40:55] = np.linspace(15, 82, 15)
    vix[55:63] = np.linspace(82, 35, 8)
    if n > 690:
        vix[500:690] = np.clip(25 + rng.normal(0, 3, 190), 18, 38)

    # SPY trend
    spy_cum = np.cumsum(spy_ret)
    trend = np.zeros(n)
    for i in range(20, n):
        trend[i] = spy_cum[i] - spy_cum[i-20]

    # Classify regimes
    regimes = []
    for i in range(n):
        v = vix[i]; t = trend[i]
        if v > 40:
            regimes.append("crash")
        elif v > 30:
            regimes.append("high_vol")
        elif v < 15 and t > 0.01:
            regimes.append("low_vol")
        elif t < -0.01 and v > 20:
            regimes.append("bear")
        elif t > 0.005:
            regimes.append("bull")
        else:
            regimes.append("bull")  # default to bull

    return {
        "strat_returns": strat_returns,
        "regimes": regimes,
        "vix": vix,
        "spy_returns": spy_ret,
        "trend": trend,
        "dates": idx,
        "n": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Backtester
# ═══════════════════════════════════════════════════════════════════════════


def _metrics(rets: np.ndarray) -> dict:
    if len(rets) < 2:
        return {"cagr": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "sortino": 0, "vol": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1/max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    ds = float(down.std()) if len(down) > 1 else std
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    return {"cagr": round(cagr*100, 2), "sharpe": round(sharpe, 2),
            "max_dd": round(dd*100, 2), "calmar": round(calmar, 2),
            "sortino": round(sortino, 2), "vol": round(std*math.sqrt(TRADING_DAYS)*100, 2)}


def _yearly(rets: np.ndarray, dates: pd.DatetimeIndex) -> Dict[int, Dict[str, float]]:
    by_yr: Dict[int, List[int]] = {}
    for i, d in enumerate(dates):
        by_yr.setdefault(d.year, []).append(i)
    out = {}
    for yr, idx in sorted(by_yr.items()):
        m = _metrics(rets[idx])
        out[yr] = {"cagr": m["cagr"], "sharpe": m["sharpe"], "dd": m["max_dd"]}
    return out


def _equity(rets: np.ndarray, capital: float = 100_000) -> List[float]:
    eq = [capital]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq


def run_method(
    name: str,
    strat_returns: Dict[str, np.ndarray],
    regimes: List[str],
    dates: pd.DatetimeIndex,
    vix: np.ndarray,
    spy_returns: np.ndarray,
    trend: np.ndarray,
) -> MethodResult:
    """Run one allocation method across the full period."""
    n = len(dates)
    port_rets = np.zeros(n)
    states: List[DayState] = []
    leverages: List[float] = []

    for i in range(n):
        regime = regimes[i]

        if name == "static":
            weights = STATIC_WEIGHTS
            leverage = STATIC_LEVERAGE
        elif name == "regime_adaptive":
            alloc = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION["bull"])
            weights = alloc["weights"]
            leverage = alloc["leverage"]
        elif name == "dynamic_sizing":
            # Use dynamic sizing logic from compass/dynamic_sizing.py
            weights = STATIC_WEIGHTS
            leverage = _dynamic_leverage(vix[i], trend[i], regime)
        else:
            weights = STATIC_WEIGHTS
            leverage = STATIC_LEVERAGE

        day_ret = 0.0
        for sid in STRATEGY_IDS:
            w = weights.get(sid, 0)
            day_ret += w * strat_returns[sid][i]
        day_ret *= leverage
        port_rets[i] = day_ret
        leverages.append(leverage)

        states.append(DayState(
            date=dates[i], regime=regime, leverage=round(leverage, 3),
            weights={k: round(v, 3) for k, v in weights.items()}, method=name))

    m = _metrics(port_rets)
    return MethodResult(
        name=name, cagr=m["cagr"], sharpe=m["sharpe"], max_dd=m["max_dd"],
        calmar=m["calmar"], sortino=m["sortino"], vol=m["vol"],
        equity=_equity(port_rets), daily_returns=port_rets,
        yearly=_yearly(port_rets, dates),
        avg_leverage=round(float(np.mean(leverages)), 3),
        states=states)


def _dynamic_leverage(vix: float, trend: float, regime: str) -> float:
    """Simplified dynamic sizing (mirrors compass/dynamic_sizing.py logic)."""
    if regime == "crash":
        return 0.5
    if vix > 30:
        return 0.5
    if vix > 25:
        return 0.8
    if vix < 15 and trend > 0.01:
        return 2.3
    if vix < 18 and trend > 0:
        return 2.0
    if trend < -0.02:
        return 0.8
    return 1.6


# ═══════════════════════════════════════════════════════════════════════════
# Comparison runner
# ═══════════════════════════════════════════════════════════════════════════


def run_comparison(seed: int = 42) -> ComparisonResult:
    """Run all three methods and compare."""
    data = generate_data(seed=seed)
    sr = data["strat_returns"]
    regimes = data["regimes"]
    dates = data["dates"]
    vix = data["vix"]
    spy = data["spy_returns"]
    trend = data["trend"]

    methods = {}
    for name in ["static", "regime_adaptive", "dynamic_sizing"]:
        methods[name] = run_method(name, sr, regimes, dates, vix, spy, trend)

    regime_dist: Dict[str, int] = {}
    for r in regimes:
        regime_dist[r] = regime_dist.get(r, 0) + 1

    winner_sharpe = max(methods, key=lambda k: methods[k].sharpe)
    winner_calmar = max(methods, key=lambda k: methods[k].calmar)

    return ComparisonResult(
        methods=methods, regime_distribution=regime_dist,
        n_days=data["n"], dates=dates,
        winner_sharpe=winner_sharpe, winner_calmar=winner_calmar)


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════


def generate_report(
    result: ComparisonResult,
    output_path: str = "reports/regime_adaptive_portfolio.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Comparison table
    headers = "".join(f"<th>{m.name.replace('_', ' ').title()}</th>" for m in result.methods.values())
    metrics_list = [
        ("CAGR", "cagr", "%", False),
        ("Sharpe", "sharpe", "", False),
        ("Max DD", "max_dd", "%", True),
        ("Calmar", "calmar", "", False),
        ("Sortino", "sortino", "", False),
        ("Vol", "vol", "%", True),
        ("Avg Leverage", "avg_leverage", "x", False),
    ]
    comp_rows = ""
    for label, attr, suffix, lower_better in metrics_list:
        vals = {name: getattr(m, attr) for name, m in result.methods.items()}
        best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        cells = ""
        for name, v in vals.items():
            bold = ' style="font-weight:700;color:#16a34a"' if name == best else ""
            cells += f"<td{bold}>{v}{suffix}</td>"
        comp_rows += f"<tr><td>{label}</td>{cells}</tr>"

    # Yearly table
    yr_rows = ""
    all_years = sorted(set(yr for m in result.methods.values() for yr in m.yearly))
    for yr in all_years:
        cells = ""
        for m in result.methods.values():
            y = m.yearly.get(yr, {"cagr": 0, "sharpe": 0, "dd": 0})
            c = "#16a34a" if y["cagr"] > 0 else "#dc2626"
            cells += f'<td style="color:{c}">{y["cagr"]:+.1f}%</td><td>{y["sharpe"]:.2f}</td><td>{y["dd"]:.1f}%</td>'
        yr_rows += f"<tr><td>{yr}</td>{cells}</tr>"

    yr_headers = "".join(
        f'<th colspan="3" style="text-align:center">{m.name.replace("_"," ").title()}</th>'
        for m in result.methods.values())
    yr_sub = "<th>CAGR</th><th>SR</th><th>DD</th>" * len(result.methods)

    # Regime allocation table
    alloc_rows = ""
    for regime, alloc in sorted(REGIME_ALLOCATION.items()):
        wt = " | ".join(f"{SHORT_NAMES[s]}:{w:.0%}" for s, w in alloc["weights"].items())
        alloc_rows += f"<tr><td>{regime}</td><td>{alloc['leverage']:.1f}x</td><td style='font-size:0.78rem'>{wt}</td></tr>"

    # Regime distribution
    dist_rows = "".join(
        f"<tr><td>{r}</td><td>{c}</td><td>{c/result.n_days*100:.1f}%</td></tr>"
        for r, c in sorted(result.regime_distribution.items(), key=lambda x: -x[1]))

    # Equity SVG
    eq_svg = _build_triple_equity_svg(result.methods)

    ws = result.winner_sharpe.replace("_", " ").title()
    wc = result.winner_calmar.replace("_", " ").title()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Regime-Adaptive Portfolio</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}td:first-child{{text-align:left}}
svg{{display:block;margin:0.5rem 0}}
.winner{{display:inline-block;padding:3px 10px;border-radius:4px;font-weight:700;font-size:0.8rem;background:#dcfce7;color:#16a34a}}
</style></head><body>
<h1>Regime-Adaptive Portfolio Allocation</h1>
<p class="meta">3-Way Comparison: Static 1.6x vs Regime-Adaptive vs Dynamic Sizing | 2020-2025 |
<span class="winner">Sharpe Winner: {ws}</span> <span class="winner">Calmar Winner: {wc}</span></p>

<div class="grid">
{"".join(f'''<div class="card"><div class="l">{m.name.replace("_"," ").title()}</div>
<div class="v">{m.sharpe:.2f} SR / {m.cagr:.0f}% / {m.max_dd:.0f}% DD</div></div>'''
for m in result.methods.values())}
</div>

<h2>Head-to-Head Comparison</h2>
<table><tr><th>Metric</th>{headers}</tr>{comp_rows}</table>

<h2>Equity Curves</h2>
{eq_svg}

<h2>Yearly Breakdown</h2>
<table><tr><th>Year</th>{yr_headers}</tr><tr><th></th>{yr_sub}</tr>{yr_rows}</table>

<h2>Regime Allocation Rules</h2>
<table><tr><th>Regime</th><th>Leverage</th><th>Weights</th></tr>{alloc_rows}</table>

<h2>Regime Distribution</h2>
<table><tr><th>Regime</th><th>Days</th><th>%</th></tr>{dist_rows}</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/regime_portfolio.py | Static vs Regime-Adaptive vs Dynamic Sizing</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


def _build_triple_equity_svg(methods: Dict[str, MethodResult]) -> str:
    colors = {"static": "#94a3b8", "regime_adaptive": "#16a34a", "dynamic_sizing": "#3b82f6"}
    labels = {"static": "Static (gray dashed)", "regime_adaptive": "Regime-Adaptive (green)",
              "dynamic_sizing": "Dynamic Sizing (blue)"}
    w, h = 780, 220
    pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w - pl - pr, h - pt - pb

    all_vals = []
    for m in methods.values():
        all_vals.extend(m.equity)
    if not all_vals:
        return ""
    ym, yx = min(all_vals) * 0.95, max(all_vals) * 1.05
    max_n = max(len(m.equity) for m in methods.values())

    paths = ""
    for name, m in methods.items():
        eq = m.equity
        n = len(eq)
        step = max(1, n // 500)
        pts = [(i, eq[i]) for i in range(0, n, step)]
        if pts[-1][0] != n - 1:
            pts.append((n - 1, eq[-1]))
        def tx(i): return pl + i / max(max_n - 1, 1) * pw
        def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph
        d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j, (i, v) in enumerate(pts))
        c = colors.get(name, "#666")
        dash = ' stroke-dasharray="4,3"' if name == "static" else ""
        paths += f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.5"{dash}/>'

    legend = " | ".join(f'<tspan fill="{colors.get(n, "#666")}">{labels.get(n, n)}</tspan>' for n in methods)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="10" fill="#64748b">{legend}</text>
  {paths}
</svg>"""


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def run_analysis(seed: int = 42) -> ComparisonResult:
    print("Regime-Adaptive Portfolio Comparison")
    print("=" * 60)

    result = run_comparison(seed=seed)

    print(f"\n  {'Method':<22} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'Calmar':>8} {'Sortino':>8} {'Avg Lev':>8}")
    print(f"  {'-'*70}")
    for m in result.methods.values():
        print(f"  {m.name:<22} {m.cagr:>7.1f}% {m.sharpe:>8.2f} {m.max_dd:>7.1f}% {m.calmar:>8.1f} {m.sortino:>8.1f} {m.avg_leverage:>7.2f}x")

    print(f"\n  Sharpe winner:  {result.winner_sharpe}")
    print(f"  Calmar winner:  {result.winner_calmar}")

    report = generate_report(result)
    print(f"\n  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
