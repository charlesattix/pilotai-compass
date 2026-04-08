"""
compass/exp2330_mc_stress_test.py — EXP-2330 Monte Carlo Stress Test.

FINAL RISK SIGN-OFF for the North Star v6 7-stream portfolio before
paper trading. Five tests:

  1. Monte Carlo bootstrap            — 10,000 paths × 5-year horizon
  2. Historical crisis replay         — 2018 volmageddon, 2020 COVID
                                          crash, 2022 bear market
  3. Correlation-stress scenario      — force all 21 pairwise
                                          correlations to +0.50 via
                                          a Cholesky factor model
  4. Regime analysis                  — bull / bear / sideways / crisis
                                          performance on the historical
                                          stream
  5. VaR / CVaR                       — 95% and 99% daily & 1-year

DATA (Rule Zero):
  compass/cache/exp2280_v6_sparse.pkl — 1566 daily return rows, 2020-01
  → 2025-12, columns: exp1220, xlf_cs, xli_cs, gld_cal, slv_cal,
  vol_arb, v5_hedge. All real IronVault + Yahoo derived streams.

WEIGHTING: equal-risk (inverse vol) scaled to a 15% target annual vol
per sleeve, then equally combined. Gross leverage clips at 3.0×.

OUTPUTS:
  compass/reports/exp2330_mc_stress_test.{json,html}
"""

from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2330_mc_stress_test.json"
REPORT_HTML = REPORT_DIR / "exp2330_mc_stress_test.html"

TRADING_DAYS = 252
TARGET_VOL_PER_SLEEVE = 0.15         # 15% annualized per sleeve
MAX_GROSS_LEVERAGE = 3.0
MC_PATHS = 10_000
MC_HORIZON_DAYS = TRADING_DAYS * 5     # 5 years
RNG_SEED = 20260407                    # deterministic for reproducibility


# ═══════════════════════════════════════════════════════════════════════════
# Load + prep streams
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioBuild:
    streams: pd.DataFrame           # raw daily returns
    weights: np.ndarray             # per-sleeve weights (post-leverage)
    stream_names: List[str]
    weighted_returns: pd.Series     # portfolio daily returns
    stream_means: np.ndarray
    stream_stds: np.ndarray
    corr_matrix: np.ndarray


def build_portfolio() -> PortfolioBuild:
    print(f"  loading {STREAMS_PKL.name} (real 7-stream cache)...")
    df: pd.DataFrame = pickle.load(STREAMS_PKL.open("rb"))
    cols = list(df.columns)
    if len(cols) != 7:
        raise RuntimeError(f"expected 7 streams, got {len(cols)}: {cols}")
    df = df.fillna(0.0).astype(float)
    print(f"    {len(df)} days × {len(cols)} streams  "
          f"{df.index.min().date()} → {df.index.max().date()}")

    # Equal-risk weights: target 15% vol per sleeve, scaled then clipped
    stds = df.std(ddof=1).values
    with np.errstate(divide="ignore", invalid="ignore"):
        per_sleeve_lev = np.where(
            stds > 1e-12,
            (TARGET_VOL_PER_SLEEVE / math.sqrt(TRADING_DAYS)) / stds,
            0.0,
        )
    # Equal blend — each sleeve gets 1/7 of the combined weight
    base_weights = per_sleeve_lev * (1.0 / len(cols))
    gross = float(np.sum(np.abs(base_weights)))
    if gross > MAX_GROSS_LEVERAGE:
        base_weights *= MAX_GROSS_LEVERAGE / gross
    weights = base_weights

    weighted = pd.Series(df.values @ weights, index=df.index, name="port")
    means = df.mean().values
    corr = df.corr().values

    print(f"    per-sleeve leverages: "
          f"{', '.join(f'{c}={w:.2f}' for c, w in zip(cols, weights))}")
    print(f"    gross leverage: {float(np.sum(np.abs(weights))):.2f}×")
    print(f"    realized portfolio vol (historical): "
          f"{float(weighted.std(ddof=1)) * math.sqrt(TRADING_DAYS)*100:.2f}%")

    return PortfolioBuild(
        streams=df,
        weights=weights,
        stream_names=cols,
        weighted_returns=weighted,
        stream_means=means,
        stream_stds=stds,
        corr_matrix=corr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Basic metrics
# ═══════════════════════════════════════════════════════════════════════════

def annualized(mu: float, sd: float) -> Tuple[float, float]:
    return mu * TRADING_DAYS, sd * math.sqrt(TRADING_DAYS)


def max_drawdown(rets: np.ndarray) -> float:
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def portfolio_metrics(rets: np.ndarray) -> Dict:
    n = len(rets)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = np.cumprod(1.0 + rets)
    yrs = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_drawdown(rets) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "hit_rate_pct": round(float((rets > 0).mean()) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1) Monte Carlo bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def mc_stationary_bootstrap(rets: np.ndarray,
                              n_paths: int,
                              horizon: int,
                              block_mean: int = 10,
                              seed: int = RNG_SEED) -> np.ndarray:
    """Politis-Romano stationary block bootstrap — preserves serial
    correlation better than plain iid resampling. Returns shape
    (n_paths, horizon)."""
    rng = np.random.default_rng(seed)
    n = len(rets)
    if n == 0:
        return np.zeros((n_paths, horizon))
    p = 1.0 / block_mean
    out = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        idx = rng.integers(0, n)
        path = np.empty(horizon)
        for t in range(horizon):
            path[t] = rets[idx]
            # stay in block with prob (1-p); otherwise jump
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = idx + 1
                if idx >= n:
                    idx = rng.integers(0, n)
        out[i] = path
    return out


def mc_summary(paths: np.ndarray) -> Dict:
    """Aggregate statistics across MC paths."""
    n_paths, horizon = paths.shape
    equities = np.cumprod(1.0 + paths, axis=1)
    terminal = equities[:, -1]
    cagrs = terminal ** (TRADING_DAYS / horizon) - 1.0
    peaks = np.maximum.accumulate(equities, axis=1)
    dd = (equities - peaks) / peaks
    max_dd = dd.min(axis=1)
    annual_vol = paths.std(axis=1, ddof=1) * math.sqrt(TRADING_DAYS)
    sharpes = np.where(
        annual_vol > 1e-12,
        paths.mean(axis=1) * TRADING_DAYS / annual_vol,
        0.0,
    )

    def q(a, p):
        return float(np.quantile(a, p))

    return {
        "n_paths": int(n_paths),
        "horizon_days": int(horizon),
        "terminal_equity": {
            "median": q(terminal, 0.50),
            "p05":    q(terminal, 0.05),
            "p25":    q(terminal, 0.25),
            "p75":    q(terminal, 0.75),
            "p95":    q(terminal, 0.95),
            "prob_loss": float((terminal < 1.0).mean()),
            "prob_doubling": float((terminal >= 2.0).mean()),
        },
        "cagr_pct": {
            "median": round(q(cagrs, 0.50) * 100, 2),
            "p05":    round(q(cagrs, 0.05) * 100, 2),
            "p95":    round(q(cagrs, 0.95) * 100, 2),
            "mean":   round(float(cagrs.mean()) * 100, 2),
        },
        "max_dd_pct": {
            "median": round(q(max_dd, 0.50) * 100, 2),
            "p05":    round(q(max_dd, 0.05) * 100, 2),  # worst 5% (most negative)
            "p25":    round(q(max_dd, 0.25) * 100, 2),
            "mean":   round(float(max_dd.mean()) * 100, 2),
            "worst":  round(float(max_dd.min()) * 100, 2),
        },
        "sharpe": {
            "median": round(q(sharpes, 0.50), 3),
            "p05":    round(q(sharpes, 0.05), 3),
            "p95":    round(q(sharpes, 0.95), 3),
            "mean":   round(float(sharpes.mean()), 3),
        },
        "breach_counts": {
            # How many MC paths breach the 12% DD ceiling at some point
            "pct_paths_breaching_12pct_dd": round(float((max_dd <= -0.12).mean()) * 100, 2),
            "pct_paths_breaching_20pct_dd": round(float((max_dd <= -0.20).mean()) * 100, 2),
            "pct_paths_losing_money":        round(float((terminal < 1.0).mean()) * 100, 2),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2) Historical crisis replay
# ═══════════════════════════════════════════════════════════════════════════

CRISIS_WINDOWS = {
    # Only 2020 COVID and 2022 bear fall inside the 2020-2025 data window.
    # 2018 volmageddon (Feb 5, 2018) is BEFORE the stream coverage starts —
    # we report it honestly as a data gap rather than extrapolating.
    "2020_covid": ("2020-02-19", "2020-04-30"),
    "2022_bear":  ("2022-01-03", "2022-10-31"),
    "2018_volmageddon": ("2018-02-01", "2018-02-16"),   # OUT OF RANGE
}


def replay_crisis(port_rets: pd.Series) -> Dict:
    out: Dict[str, Dict] = {}
    for name, (start, end) in CRISIS_WINDOWS.items():
        window = port_rets[(port_rets.index >= start) & (port_rets.index <= end)]
        if len(window) < 5:
            out[name] = {
                "window": f"{start} → {end}",
                "status": "DATA_GAP",
                "n_days": int(len(window)),
                "note": ("Window lies outside the 2020-2025 stream coverage; "
                          "not extrapolated. No synthetic fill (Rule Zero)."),
            }
            continue
        m = portfolio_metrics(window.values)
        out[name] = {
            "window": f"{start} → {end}",
            "status": "OK",
            **m,
            "total_return_pct": round((float(np.prod(1.0 + window.values)) - 1.0) * 100, 3),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3) Correlation stress
# ═══════════════════════════════════════════════════════════════════════════

def correlation_stress(build: PortfolioBuild, forced_corr: float = 0.50,
                          n_paths: int = 2000, horizon: int = TRADING_DAYS,
                          seed: int = RNG_SEED + 1) -> Dict:
    """Simulate MC paths where ALL pairwise correlations are forced to
    `forced_corr` (0.50 by default). Per-stream marginals (mean, std)
    are preserved from empirical estimates."""
    rng = np.random.default_rng(seed)
    n_streams = len(build.stream_names)

    # Forced correlation matrix
    C = np.full((n_streams, n_streams), forced_corr)
    np.fill_diagonal(C, 1.0)
    try:
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n_streams))
    except np.linalg.LinAlgError:
        # Fallback: eigenvalue clip
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, 1e-6)
        L = evecs @ np.diag(np.sqrt(evals))

    means = build.stream_means
    stds = build.stream_stds

    paths = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        z = rng.standard_normal((horizon, n_streams))
        correlated = z @ L.T
        # Apply marginals
        stream_rets = correlated * stds + means
        port_daily = stream_rets @ build.weights
        paths[i] = port_daily

    # Compare against the baseline correlation MC (empirical corr)
    base_paths = correlation_stress_baseline(build, n_paths, horizon, seed + 100)

    def summarise(p: np.ndarray) -> Dict:
        eq = np.cumprod(1.0 + p, axis=1)
        max_dd = ((eq - np.maximum.accumulate(eq, axis=1)) / np.maximum.accumulate(eq, axis=1)).min(axis=1)
        ann_vol = p.std(axis=1, ddof=1) * math.sqrt(TRADING_DAYS)
        ann_mu = p.mean(axis=1) * TRADING_DAYS
        sharpes = np.where(ann_vol > 1e-12, ann_mu / ann_vol, 0.0)
        return {
            "median_cagr_pct": round(float(np.median(ann_mu)) * 100, 2),
            "median_vol_pct": round(float(np.median(ann_vol)) * 100, 2),
            "median_sharpe": round(float(np.median(sharpes)), 3),
            "p05_max_dd_pct": round(float(np.quantile(max_dd, 0.05)) * 100, 2),
            "median_max_dd_pct": round(float(np.median(max_dd)) * 100, 2),
            "worst_max_dd_pct": round(float(max_dd.min()) * 100, 2),
            "pct_breaching_12_dd": round(float((max_dd <= -0.12).mean()) * 100, 2),
        }

    return {
        "forced_corr": forced_corr,
        "n_paths": n_paths,
        "horizon_days": horizon,
        "empirical_correlation": summarise(base_paths),
        "forced_correlation": summarise(paths),
    }


def correlation_stress_baseline(build: PortfolioBuild, n_paths: int,
                                  horizon: int, seed: int) -> np.ndarray:
    """Same MC structure but with empirical correlation preserved."""
    rng = np.random.default_rng(seed)
    n_streams = len(build.stream_names)
    C = build.corr_matrix.copy()
    try:
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n_streams))
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, 1e-6)
        L = evecs @ np.diag(np.sqrt(evals))
    paths = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        z = rng.standard_normal((horizon, n_streams))
        corr = z @ L.T
        stream_rets = corr * build.stream_stds + build.stream_means
        paths[i] = stream_rets @ build.weights
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# 4) Regime analysis — bull / bear / sideways / crisis
# ═══════════════════════════════════════════════════════════════════════════

def classify_regimes(spy_rets: pd.Series) -> pd.Series:
    """Simple regime classifier on SPY daily returns:
      - CRISIS : 60-day rolling vol > 30% AND rolling max DD > 10%
      - BEAR   : rolling 60d return < -5% (and not crisis)
      - BULL   : rolling 60d return > +5%
      - SIDEWAYS : the rest
    """
    roll = spy_rets.rolling(60, min_periods=20)
    ann_vol = roll.std() * math.sqrt(TRADING_DAYS)
    roll_ret = (1.0 + spy_rets).rolling(60, min_periods=20).apply(
        lambda r: float(np.prod(r) - 1.0), raw=True
    )
    eq = (1.0 + spy_rets).cumprod()
    peak60 = eq.rolling(60, min_periods=20).max()
    dd60 = (eq - peak60) / peak60

    regime = pd.Series("SIDEWAYS", index=spy_rets.index)
    regime[(ann_vol > 0.30) & (dd60 < -0.10)] = "CRISIS"
    regime[(regime != "CRISIS") & (roll_ret < -0.05)] = "BEAR"
    regime[(regime != "CRISIS") & (regime != "BEAR") & (roll_ret > 0.05)] = "BULL"
    return regime


def regime_analysis(build: PortfolioBuild) -> Dict:
    import urllib.request
    # Pull real SPY from Yahoo for regime classification
    start_ts = int(pd.Timestamp("2019-12-01").timestamp())
    end_ts = int(pd.Timestamp("2026-01-01").timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = pd.Series(
        result["indicators"]["quote"][0]["close"],
        index=pd.DatetimeIndex([datetime.fromtimestamp(t).date() for t in result["timestamp"]]),
    ).dropna()
    spy_rets = closes.pct_change().dropna()
    regime = classify_regimes(spy_rets)

    # Align to portfolio dates
    port = build.weighted_returns
    idx = port.index.intersection(regime.index)
    port_aligned = port.reindex(idx)
    regime_aligned = regime.reindex(idx)

    out: Dict[str, Dict] = {}
    for r in ["BULL", "BEAR", "SIDEWAYS", "CRISIS"]:
        mask = regime_aligned == r
        n = int(mask.sum())
        if n < 5:
            out[r] = {"n_days": n, "status": "too few days"}
            continue
        sub = port_aligned[mask].values
        m = portfolio_metrics(sub)
        out[r] = {
            "n_days": n,
            **m,
            "share_of_sample_pct": round(n / len(regime_aligned) * 100, 2),
        }
    return {
        "regime_dates": {
            "start": str(idx.min().date()),
            "end": str(idx.max().date()),
            "total_days": int(len(idx)),
        },
        "by_regime": out,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5) VaR / CVaR
# ═══════════════════════════════════════════════════════════════════════════

def var_cvar(rets: np.ndarray) -> Dict:
    """Historical VaR and CVaR at 95% and 99%, daily and 1-year scaled."""
    r = np.sort(rets)
    n = len(r)
    if n < 20:
        return {}

    def lookup(alpha: float) -> Tuple[float, float]:
        k = int((1 - alpha) * n)
        var_ = r[k]
        cvar = float(r[: max(1, k)].mean())
        return var_, cvar

    var95_d, cvar95_d = lookup(0.95)
    var99_d, cvar99_d = lookup(0.99)

    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    # Parametric (Gaussian) 1-year VaR via SQRT-t scaling
    from math import sqrt
    z95, z99 = 1.645, 2.326
    mu_yr = mu * TRADING_DAYS
    sd_yr = sd * sqrt(TRADING_DAYS)
    var95_yr = mu_yr - z95 * sd_yr
    var99_yr = mu_yr - z99 * sd_yr
    # Parametric CVaR: -φ(z)/(1-α) * σ + μ (Gaussian)
    from math import pi, exp
    def pdf(z): return exp(-0.5 * z * z) / math.sqrt(2 * pi)
    cvar95_yr = mu_yr - (pdf(z95) / 0.05) * sd_yr
    cvar99_yr = mu_yr - (pdf(z99) / 0.01) * sd_yr

    return {
        "historical_daily": {
            "var_95_pct": round(var95_d * 100, 3),
            "cvar_95_pct": round(cvar95_d * 100, 3),
            "var_99_pct": round(var99_d * 100, 3),
            "cvar_99_pct": round(cvar99_d * 100, 3),
        },
        "parametric_annual": {
            "var_95_pct": round(var95_yr * 100, 3),
            "cvar_95_pct": round(cvar95_yr * 100, 3),
            "var_99_pct": round(var99_yr * 100, 3),
            "cvar_99_pct": round(cvar99_yr * 100, 3),
        },
        "n_observations": int(n),
        "worst_day_pct": round(float(r[0]) * 100, 3),
        "best_day_pct": round(float(r[-1]) * 100, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    mc = payload["monte_carlo"]
    vc = payload["var_cvar"]
    crisis = payload["crisis_replay"]
    corr = payload["correlation_stress"]
    reg = payload["regime_analysis"]["by_regime"]
    sign = payload["sign_off"]

    crisis_rows = ""
    for name, c in crisis.items():
        if c.get("status") == "DATA_GAP":
            crisis_rows += f"""<tr>
                <td><strong>{name}</strong></td>
                <td>{c['window']}</td>
                <td colspan="5" class="warn">{c['note']}</td>
            </tr>"""
        else:
            cagr_cls = "good" if c["cagr_pct"] > 0 else "bad"
            crisis_rows += f"""<tr>
                <td><strong>{name}</strong></td>
                <td>{c['window']}</td>
                <td>{c['n_days']}</td>
                <td class="{cagr_cls}">{c['total_return_pct']:+.2f}%</td>
                <td>{c['sharpe']:.2f}</td>
                <td class="bad">{c['max_dd_pct']:.2f}%</td>
                <td>{c['vol_pct']:.2f}%</td>
            </tr>"""

    reg_rows = ""
    for name, r in reg.items():
        if "cagr_pct" not in r:
            reg_rows += f"""<tr><td><strong>{name}</strong></td>
                <td colspan="6">{r.get('status','?')}</td></tr>"""
            continue
        cls = "good" if r["cagr_pct"] > 0 else "bad"
        reg_rows += f"""<tr>
            <td><strong>{name}</strong></td>
            <td>{r['n_days']}</td>
            <td>{r['share_of_sample_pct']:.1f}%</td>
            <td class="{cls}">{r['cagr_pct']:.2f}%</td>
            <td>{r['sharpe']:.2f}</td>
            <td>{r['max_dd_pct']:.2f}%</td>
            <td>{r['vol_pct']:.2f}%</td>
        </tr>"""

    sign_cls = "good" if sign["status"] == "APPROVE" else "bad"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2330 North Star v6 MC Stress Test</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:140px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  .warn {{ color:#ca8a04; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.84em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.72em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; font-size:0.9em; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2330 — Monte Carlo Stress Test · North Star v6</h1>
<div class="subtitle">Final risk sign-off before paper trading | {payload['timestamp']}</div>

<div class="note">
    <strong>Portfolio:</strong> 7-stream North Star v6 (exp1220, xlf_cs,
    xli_cs, gld_cal, slv_cal, vol_arb, v5_hedge) at equal-risk 15%/sleeve
    target vol, gross leverage clipped at 3.0×. <strong>Data:</strong>
    1566 daily rows (2020-01 → 2025-12) from compass/cache/exp2280_v6_sparse.pkl,
    all real IronVault + Yahoo derived. <strong>MC method:</strong>
    Politis-Romano stationary bootstrap (block mean 10 days) preserves
    serial correlation; 10,000 paths × 5-year horizon.
</div>

<h2>Sign-Off Decision</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {sign_cls}">{sign['status']}</div><div class="label">Decision</div></div>
    <div class="kpi"><div class="value">{mc['cagr_pct']['median']:.1f}%</div><div class="label">MC Median CAGR</div></div>
    <div class="kpi"><div class="value">{mc['max_dd_pct']['p05']:.1f}%</div><div class="label">MC 5% DD Tail</div></div>
    <div class="kpi"><div class="value">{mc['breach_counts']['pct_paths_breaching_12pct_dd']:.1f}%</div><div class="label">Paths ≥12% DD</div></div>
    <div class="kpi"><div class="value">{mc['breach_counts']['pct_paths_losing_money']:.1f}%</div><div class="label">Paths Losing $</div></div>
</div>

<h2>1) Monte Carlo Bootstrap — 10,000 paths × 5-year horizon</h2>
<table>
    <thead><tr><th>Metric</th><th>p05</th><th>p25</th><th>median</th><th>mean</th><th>p75</th><th>p95</th></tr></thead>
    <tbody>
        <tr><td><strong>Terminal equity ($1 start)</strong></td>
            <td>{mc['terminal_equity']['p05']:.2f}</td>
            <td>{mc['terminal_equity']['p25']:.2f}</td>
            <td>{mc['terminal_equity']['median']:.2f}</td>
            <td>—</td>
            <td>{mc['terminal_equity']['p75']:.2f}</td>
            <td>{mc['terminal_equity']['p95']:.2f}</td></tr>
        <tr><td><strong>CAGR</strong></td>
            <td>{mc['cagr_pct']['p05']:.2f}%</td>
            <td>—</td>
            <td>{mc['cagr_pct']['median']:.2f}%</td>
            <td>{mc['cagr_pct']['mean']:.2f}%</td>
            <td>—</td>
            <td>{mc['cagr_pct']['p95']:.2f}%</td></tr>
        <tr><td><strong>Max DD</strong></td>
            <td class="bad">{mc['max_dd_pct']['p05']:.2f}%</td>
            <td>{mc['max_dd_pct']['p25']:.2f}%</td>
            <td>{mc['max_dd_pct']['median']:.2f}%</td>
            <td>{mc['max_dd_pct']['mean']:.2f}%</td>
            <td>—</td>
            <td>—</td></tr>
        <tr><td><strong>Sharpe</strong></td>
            <td>{mc['sharpe']['p05']:.2f}</td>
            <td>—</td>
            <td>{mc['sharpe']['median']:.2f}</td>
            <td>{mc['sharpe']['mean']:.2f}</td>
            <td>—</td>
            <td>{mc['sharpe']['p95']:.2f}</td></tr>
    </tbody>
</table>
<p class="note">
    <strong>Breach counts:</strong>
    {mc['breach_counts']['pct_paths_breaching_12pct_dd']:.2f}% of paths breach the 12% DD ceiling at some point over 5 years,
    {mc['breach_counts']['pct_paths_breaching_20pct_dd']:.2f}% breach 20%,
    and {mc['breach_counts']['pct_paths_losing_money']:.2f}% end up below starting capital.
</p>

<h2>2) Historical Crisis Replay</h2>
<table>
    <thead><tr><th>Event</th><th>Window</th><th>Days</th><th>Total Return</th>
    <th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
    <tbody>{crisis_rows}</tbody>
</table>

<h2>3) Correlation Stress — Force ρ = 0.50 on all pairs</h2>
<table>
    <thead><tr><th>Scenario</th><th>Median CAGR</th><th>Median Vol</th>
    <th>Median Sharpe</th><th>5% DD Tail</th><th>Worst DD</th><th>% ≥12% DD</th></tr></thead>
    <tbody>
        <tr><td>Empirical corr</td>
            <td>{corr['empirical_correlation']['median_cagr_pct']:.2f}%</td>
            <td>{corr['empirical_correlation']['median_vol_pct']:.2f}%</td>
            <td>{corr['empirical_correlation']['median_sharpe']:.2f}</td>
            <td class="bad">{corr['empirical_correlation']['p05_max_dd_pct']:.2f}%</td>
            <td class="bad">{corr['empirical_correlation']['worst_max_dd_pct']:.2f}%</td>
            <td>{corr['empirical_correlation']['pct_breaching_12_dd']:.2f}%</td></tr>
        <tr><td>Forced ρ = 0.50</td>
            <td>{corr['forced_correlation']['median_cagr_pct']:.2f}%</td>
            <td>{corr['forced_correlation']['median_vol_pct']:.2f}%</td>
            <td>{corr['forced_correlation']['median_sharpe']:.2f}</td>
            <td class="bad">{corr['forced_correlation']['p05_max_dd_pct']:.2f}%</td>
            <td class="bad">{corr['forced_correlation']['worst_max_dd_pct']:.2f}%</td>
            <td>{corr['forced_correlation']['pct_breaching_12_dd']:.2f}%</td></tr>
    </tbody>
</table>

<h2>4) Regime Analysis</h2>
<table>
    <thead><tr><th>Regime</th><th>Days</th><th>Share</th><th>CAGR</th>
    <th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
    <tbody>{reg_rows}</tbody>
</table>

<h2>5) VaR / CVaR</h2>
<table>
    <thead><tr><th>Metric</th><th>95%</th><th>99%</th></tr></thead>
    <tbody>
        <tr><td>Historical daily VaR</td>
            <td class="bad">{vc['historical_daily']['var_95_pct']:.2f}%</td>
            <td class="bad">{vc['historical_daily']['var_99_pct']:.2f}%</td></tr>
        <tr><td>Historical daily CVaR (expected shortfall)</td>
            <td class="bad">{vc['historical_daily']['cvar_95_pct']:.2f}%</td>
            <td class="bad">{vc['historical_daily']['cvar_99_pct']:.2f}%</td></tr>
        <tr><td>Parametric annual VaR (Gaussian SQRT-t)</td>
            <td class="bad">{vc['parametric_annual']['var_95_pct']:.2f}%</td>
            <td class="bad">{vc['parametric_annual']['var_99_pct']:.2f}%</td></tr>
        <tr><td>Parametric annual CVaR</td>
            <td class="bad">{vc['parametric_annual']['cvar_95_pct']:.2f}%</td>
            <td class="bad">{vc['parametric_annual']['cvar_99_pct']:.2f}%</td></tr>
    </tbody>
</table>

<h2>Sign-Off Reasoning</h2>
<ul>
    {''.join(f'<li>{r}</li>' for r in sign['reasons'])}
</ul>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2330 — compass/exp2330_mc_stress_test.py · Real 7-stream cache · Deterministic seed {RNG_SEED}
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Sign-off logic
# ═══════════════════════════════════════════════════════════════════════════

def sign_off(mc: Dict, crisis: Dict, corr: Dict, vc: Dict) -> Dict:
    """APPROVE the portfolio for paper if:
      (a) <10% of MC paths breach the 12% DD ceiling,
      (b) MC p05 max_dd ≥ -20% (worst 5% is still bounded),
      (c) 2020 COVID replay total return ≥ -15%,
      (d) 2022 bear total return ≥ -15%,
      (e) Forced-0.50 correlation still keeps median DD ≤ -25%,
      (f) 99% daily VaR ≤ 4% (notional).
    """
    reasons: List[str] = []
    gates: List[bool] = []

    g_a = mc["breach_counts"]["pct_paths_breaching_12pct_dd"] < 10.0
    gates.append(g_a)
    reasons.append(
        f"(a) {mc['breach_counts']['pct_paths_breaching_12pct_dd']:.2f}% MC paths breach 12% DD — "
        f"{'PASS' if g_a else 'FAIL'} (<10% target)")

    g_b = mc["max_dd_pct"]["p05"] >= -20.0
    gates.append(g_b)
    reasons.append(
        f"(b) MC p05 (worst-5%) max DD = {mc['max_dd_pct']['p05']:.2f}% — "
        f"{'PASS' if g_b else 'FAIL'} (≥ -20% target)")

    c2020 = crisis.get("2020_covid", {})
    if c2020.get("status") == "OK":
        g_c = c2020["total_return_pct"] >= -15.0
        gates.append(g_c)
        reasons.append(
            f"(c) 2020 COVID replay total = {c2020['total_return_pct']:+.2f}% — "
            f"{'PASS' if g_c else 'FAIL'} (≥ -15% target)")
    else:
        reasons.append("(c) 2020 COVID replay: data gap / skipped")

    c2022 = crisis.get("2022_bear", {})
    if c2022.get("status") == "OK":
        g_d = c2022["total_return_pct"] >= -15.0
        gates.append(g_d)
        reasons.append(
            f"(d) 2022 bear replay total = {c2022['total_return_pct']:+.2f}% — "
            f"{'PASS' if g_d else 'FAIL'} (≥ -15% target)")
    else:
        reasons.append("(d) 2022 bear replay: data gap / skipped")

    g_e = corr["forced_correlation"]["median_max_dd_pct"] >= -25.0
    gates.append(g_e)
    reasons.append(
        f"(e) forced-ρ 0.50 median DD = {corr['forced_correlation']['median_max_dd_pct']:.2f}% — "
        f"{'PASS' if g_e else 'FAIL'} (≥ -25% target)")

    var99 = abs(vc["historical_daily"]["var_99_pct"])
    g_f = var99 <= 4.0
    gates.append(g_f)
    reasons.append(
        f"(f) 99% daily historical VaR = {var99:.2f}% — "
        f"{'PASS' if g_f else 'FAIL'} (≤ 4% target)")

    ok = all(gates)
    return {
        "status": "APPROVE" if ok else "REJECT",
        "gates_passed": int(sum(gates)),
        "gates_total": len(gates),
        "reasons": reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2330 — Monte Carlo Stress Test · North Star v6")
    print("=" * 72)

    print("\n[1/6] Building 7-stream portfolio (equal-risk 15%/sleeve)...")
    build = build_portfolio()
    hist_metrics = portfolio_metrics(build.weighted_returns.values)
    print(f"  historical portfolio: CAGR={hist_metrics['cagr_pct']}%  "
          f"Sharpe={hist_metrics['sharpe']}  DD={hist_metrics['max_dd_pct']}%")

    print("\n[2/6] Monte Carlo (10,000 paths × 5-year horizon, stationary bootstrap)...")
    paths = mc_stationary_bootstrap(
        build.weighted_returns.values, n_paths=MC_PATHS, horizon=MC_HORIZON_DAYS
    )
    mc = mc_summary(paths)
    print(f"  median CAGR {mc['cagr_pct']['median']}%  "
          f"p05 CAGR {mc['cagr_pct']['p05']}%  "
          f"p95 CAGR {mc['cagr_pct']['p95']}%")
    print(f"  median Sharpe {mc['sharpe']['median']}  "
          f"p05 Sharpe {mc['sharpe']['p05']}  "
          f"p95 Sharpe {mc['sharpe']['p95']}")
    print(f"  worst DD {mc['max_dd_pct']['worst']}%  "
          f"p05 DD {mc['max_dd_pct']['p05']}%  "
          f"median DD {mc['max_dd_pct']['median']}%")
    print(f"  {mc['breach_counts']['pct_paths_breaching_12pct_dd']}% paths breach 12% DD ceiling")
    print(f"  {mc['breach_counts']['pct_paths_losing_money']}% paths finish below starting capital")

    print("\n[3/6] Historical crisis replay...")
    crisis = replay_crisis(build.weighted_returns)
    for name, c in crisis.items():
        if c.get("status") == "OK":
            print(f"  {name:18s} {c['window']}  "
                  f"ret={c['total_return_pct']:+.2f}%  "
                  f"Sh={c['sharpe']:.2f}  DD={c['max_dd_pct']:.2f}%")
        else:
            print(f"  {name:18s} {c['window']}  DATA_GAP")

    print("\n[4/6] Correlation stress (forcing all pairs to ρ=0.50)...")
    corr = correlation_stress(build)
    emp = corr["empirical_correlation"]
    frc = corr["forced_correlation"]
    print(f"  empirical:  CAGR={emp['median_cagr_pct']}%  "
          f"Sh={emp['median_sharpe']}  "
          f"p05 DD={emp['p05_max_dd_pct']}%  "
          f"breach12={emp['pct_breaching_12_dd']}%")
    print(f"  forced 0.5: CAGR={frc['median_cagr_pct']}%  "
          f"Sh={frc['median_sharpe']}  "
          f"p05 DD={frc['p05_max_dd_pct']}%  "
          f"breach12={frc['pct_breaching_12_dd']}%")

    print("\n[5/6] Regime analysis (SPY-classified bull/bear/sideways/crisis)...")
    reg = regime_analysis(build)
    for name, r in reg["by_regime"].items():
        if "cagr_pct" in r:
            print(f"  {name:9s} n={r['n_days']:4d} ({r['share_of_sample_pct']:.1f}%)  "
                  f"CAGR={r['cagr_pct']:.2f}%  Sh={r['sharpe']:.2f}  "
                  f"DD={r['max_dd_pct']:.2f}%")
        else:
            print(f"  {name:9s} {r.get('status','?')}")

    print("\n[6/6] VaR / CVaR...")
    vc = var_cvar(build.weighted_returns.values)
    hd = vc["historical_daily"]
    pa = vc["parametric_annual"]
    print(f"  Historical daily: VaR95 {hd['var_95_pct']}%  "
          f"CVaR95 {hd['cvar_95_pct']}%  "
          f"VaR99 {hd['var_99_pct']}%  CVaR99 {hd['cvar_99_pct']}%")
    print(f"  Parametric annual: VaR95 {pa['var_95_pct']}%  "
          f"CVaR95 {pa['cvar_95_pct']}%  "
          f"VaR99 {pa['var_99_pct']}%  CVaR99 {pa['cvar_99_pct']}%")

    sign = sign_off(mc, crisis, corr, vc)
    print(f"\n  SIGN-OFF: {sign['status']}  "
          f"({sign['gates_passed']}/{sign['gates_total']} gates passed)")
    for r in sign["reasons"]:
        print(f"    • {r}")

    payload = {
        "experiment": "EXP-2330",
        "title": "Monte Carlo Stress Test · North Star v6",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "portfolio": {
            "streams": build.stream_names,
            "per_sleeve_leverages": {
                name: round(float(w), 4)
                for name, w in zip(build.stream_names, build.weights)
            },
            "gross_leverage": round(float(np.sum(np.abs(build.weights))), 3),
            "target_vol_per_sleeve_pct": TARGET_VOL_PER_SLEEVE * 100,
            "max_gross_leverage": MAX_GROSS_LEVERAGE,
            "historical_metrics": hist_metrics,
        },
        "data": {
            "source": "compass/cache/exp2280_v6_sparse.pkl",
            "n_days": int(len(build.weighted_returns)),
            "start": str(build.weighted_returns.index.min().date()),
            "end": str(build.weighted_returns.index.max().date()),
        },
        "monte_carlo": mc,
        "crisis_replay": crisis,
        "correlation_stress": corr,
        "regime_analysis": reg,
        "var_cvar": vc,
        "sign_off": sign,
        "mc_config": {
            "method": "Politis-Romano stationary bootstrap (block mean 10 days)",
            "n_paths": MC_PATHS,
            "horizon_days": MC_HORIZON_DAYS,
            "seed": RNG_SEED,
        },
        "rule_zero": (
            "All returns are real (EXP-1220 canonical, EXP-1770 walk-forward "
            "GLD/SLV, XLF/XLI credit spreads, vol_arb, Crisis Alpha v5). MC "
            "samples bootstrap FROM the real returns — no synthetic "
            "distribution fitting. Correlation stress uses a factor model "
            "with real marginals and an imposed correlation matrix. Regime "
            "labels come from real Yahoo SPY data."
        ),
    }

    print("\nWriting reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
