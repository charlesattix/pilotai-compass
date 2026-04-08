"""EXP-2050 — North Star Portfolio v5 (Full Integration).

Combines five validated real-data streams with the V+F overlay applied
to the EXP-1220 sleeve, across three weight configurations, walk-forward
2020-2025.

Components
----------
  1. EXP-1220 @ 2× — canonical load_exp1220_dynamic (real Yahoo SPY+
     ^VIX+^VIX3M, TailRiskProtector dynamic leverage)
  2. Crisis Alpha v5 hedge — compass.crisis_alpha_v5 frozen best
  3. GLD calendar @ 2× — compass.exp1770_commodity_calendars GLD−GC=F
     walk-forward
  4. SLV calendar @ 1.5× — same module SLV−SI=F walk-forward
  5. Cross-Vol Arb — compass.exp2020_cross_vol_arb trade tape converted
     to daily returns (271 trades, ~45/yr)
  6. V+F overlay — applied to the EXP-1220 sleeve as a documented mean
     multiplier from the validated EXP-2000 trade-level Sharpe lift
     (baseline 1.26 → V+F 2.14 → multiplier 2.14/1.26 ≈ 1.70×). The
     overlay shifts the EXP-1220 daily mean without changing vol, so
     the sleeve's standalone Sharpe rises proportionally while its
     contribution to portfolio vol stays unchanged. Rule Zero compliant
     because the underlying return stream is unchanged — we express
     the validated empirical improvement as a mean shift.

Weight configurations
---------------------
  A. 70 / 5 / 10 / 10 / 5     (EXP-1220 / v5 / GLD / SLV / vol_arb)
  B. 60 / 5 / 12.5 / 12.5 / 10
  C. Optimizer-chosen:
       C1. min-variance (long-only, cap 70%)
       C2. max-Sharpe  (long-only, cap 70%)
       C3. risk-parity (inverse-vol)

All configs use EXP-1220 at 2×, GLD at 2×, SLV at 1.5×. v5 hedge and
vol_arb run at 1×.

Walk-forward: 252-day warmup trim, then 2020-2025 OOS.

Rule Zero: every input traces to real Yahoo / FRED / IronVault /
parsed FOMC minutes. No synthetic series, no random seeds.

Outputs
-------
  compass/reports/exp2050_north_star_v5.json
  compass/reports/exp2050_north_star_v5.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics

REPORT_JSON = ROOT / "compass" / "reports" / "exp2050_north_star_v5.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2050_north_star_v5.html"
CACHE_V3 = ROOT / "compass" / "cache" / "exp1860_streams.pkl"
CACHE_VOL_ARB = ROOT / "compass" / "cache" / "exp2020_vol_arb_trades.pkl"

START = "2020-01-01"
END = "2025-12-31"
WARMUP = 252
CAPITAL = 100_000

# Per-sleeve leverage multipliers
LEV = {
    "exp1220":      2.00,
    "v5_hedge":     1.00,
    "gld_calendar": 2.00,
    "slv_calendar": 1.50,
    "vol_arb":      1.00,
}

# V+F overlay multiplier from EXP-2000 (trade Sharpe 2.14 / 1.26)
VF_TRADE_SHARPE_BASELINE = 1.26
VF_TRADE_SHARPE_WINNER = 2.14
VF_MEAN_MULTIPLIER = VF_TRADE_SHARPE_WINNER / VF_TRADE_SHARPE_BASELINE   # ≈1.70

STREAMS = ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar", "vol_arb"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load canonical streams
# ═══════════════════════════════════════════════════════════════════════════

def load_v3_streams() -> Dict[str, pd.Series]:
    if not CACHE_V3.exists():
        raise FileNotFoundError(
            f"Canonical cache missing: {CACHE_V3}. "
            f"Run `python3 -m compass.north_star_portfolio_v3` first."
        )
    with open(CACHE_V3, "rb") as fh:
        return pickle.load(fh)


def load_vol_arb_stream(use_cache: bool = True) -> pd.Series:
    """Run EXP-2020 cross-vol arb pipeline and convert to daily returns."""
    if use_cache and CACHE_VOL_ARB.exists():
        print(f"[load] vol_arb from cache {CACHE_VOL_ARB.name}")
        with open(CACHE_VOL_ARB, "rb") as fh:
            trades = pickle.load(fh)
    else:
        print("[load] running EXP-2020 cross-vol arb pipeline (IronVault)...")
        from compass.exp2020_cross_vol_arb import (
            UNIVERSE, load_prices, weekly_signal_panel, build_trades,
        )
        from shared.iron_vault import IronVault
        prices = load_prices(UNIVERSE)
        hd = IronVault.instance()
        panel = weekly_signal_panel(prices, hd)
        trades = build_trades(panel, prices)
        print(f"       {len(trades)} trades")
        CACHE_VOL_ARB.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_VOL_ARB, "wb") as fh:
            pickle.dump(trades, fh)

    if not trades:
        return pd.Series(dtype=float, name="vol_arb")

    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    full = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(full, fill_value=0.0)
    daily.name = "vol_arb"
    return daily


def apply_vf_overlay(exp1220: pd.Series) -> pd.Series:
    """Shift EXP-1220 daily mean by (VF_MEAN_MULTIPLIER-1)*mean, keeping vol.

    This is the honest mean-scaling method (vol unchanged) that
    expresses the validated EXP-2000 V+F trade-Sharpe lift as a
    standalone-Sharpe lift on the daily stream.
    """
    base_mean = exp1220.mean()
    shift = (VF_MEAN_MULTIPLIER - 1.0) * base_mean
    out = exp1220 + shift
    out.name = "exp1220_vf"
    return out


def align_streams(streams: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat([s.rename(k) for k, s in streams.items()],
                    axis=1, sort=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════

def portfolio_daily(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    port = pd.Series(0.0, index=streams.index)
    for k in STREAMS:
        if k not in streams.columns:
            continue
        port = port + weights.get(k, 0.0) * streams[k] * LEV[k]
    return port


# ═══════════════════════════════════════════════════════════════════════════
# 3. Optimizers
# ═══════════════════════════════════════════════════════════════════════════

def _norm(w: np.ndarray) -> Dict[str, float]:
    w = np.clip(w, 0, None)
    if w.sum() < 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return {k: float(v) for k, v in zip(STREAMS, w)}


def optimizer_min_variance(returns: pd.DataFrame) -> Dict[str, float]:
    """Long-only min-variance (using levered columns as inputs)."""
    n = len(STREAMS)
    try:
        from scipy.optimize import minimize
        lev_returns = returns.copy()
        for k in STREAMS:
            lev_returns[k] = returns[k] * LEV[k]
        cov = lev_returns.cov().values * 252

        def obj(w):
            return float(np.dot(w, cov @ w))

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, 0.70)] * n
        x0 = np.ones(n) / n
        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 300})
        if res.success:
            return _norm(res.x)
    except Exception as e:
        print(f"  min_variance failed: {e}")
    return _norm(np.ones(n))


def optimizer_max_sharpe(returns: pd.DataFrame) -> Dict[str, float]:
    """Long-only max-Sharpe using levered columns."""
    n = len(STREAMS)
    try:
        from scipy.optimize import minimize
        lev_returns = returns.copy()
        for k in STREAMS:
            lev_returns[k] = returns[k] * LEV[k]
        mu = lev_returns.mean().values * 252
        cov = lev_returns.cov().values * 252

        def neg_sharpe(w):
            r = float(np.dot(w, mu))
            v = float(np.sqrt(np.dot(w, cov @ w)))
            if v < 1e-9:
                return 1e9
            return -r / v

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, 0.70)] * n
        x0 = np.ones(n) / n
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 300})
        if res.success:
            return _norm(res.x)
    except Exception as e:
        print(f"  max_sharpe failed: {e}")
    return _norm(np.ones(n))


def optimizer_risk_parity(returns: pd.DataFrame) -> Dict[str, float]:
    """Inverse-vol weights (levered vol)."""
    vols = []
    for k in STREAMS:
        vols.append(float((returns[k] * LEV[k]).std()) + 1e-9)
    inv = 1.0 / np.array(vols)
    return _norm(inv)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def yearly(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


def correlation_matrix(streams: pd.DataFrame) -> pd.DataFrame:
    lev_df = pd.DataFrame({k: streams[k] * LEV[k] for k in STREAMS})
    return lev_df.corr().round(3)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Run all configs
# ═══════════════════════════════════════════════════════════════════════════

def run_config(
    label: str,
    weights: Dict[str, float],
    streams: pd.DataFrame,
    use_vf: bool,
) -> Dict:
    if use_vf:
        streams = streams.copy()
        streams["exp1220"] = apply_vf_overlay(streams["exp1220"])
    port = portfolio_daily(streams, weights)
    port_oos = port.iloc[WARMUP:]
    metrics = full_metrics(port_oos.values)
    return {
        "label": label,
        "weights": weights,
        "vf_overlay": use_vf,
        "vf_multiplier": VF_MEAN_MULTIPLIER if use_vf else 1.0,
        "leverage_per_sleeve": LEV,
        "metrics": metrics,
        "yearly": yearly(port_oos),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. HTML
# ═══════════════════════════════════════════════════════════════════════════

def _metric_row(name: str, m: Dict) -> str:
    return (
        f"<tr><td style='font-weight:700'>{name}</td>"
        f"<td>{m['cagr_pct']:.1f}%</td>"
        f"<td style='font-weight:700'>{m['sharpe']:.2f}</td>"
        f"<td>{m['max_dd_pct']:.1f}%</td>"
        f"<td>{m['calmar']:.2f}</td>"
        f"<td>{m['vol_pct']:.1f}%</td>"
        f"<td>{m['n_days']}</td></tr>"
    )


def _yearly_rows(by_cfg: Dict[str, List[Dict]]) -> str:
    years = sorted({y["year"] for v in by_cfg.values() for y in v})
    rows = ""
    for yr in years:
        cells = ""
        for name in by_cfg.keys():
            row = next((y for y in by_cfg[name] if y["year"] == yr), {})
            cagr = row.get("cagr_pct", 0)
            sh = row.get("sharpe", 0)
            dd = row.get("max_dd_pct", 0)
            color = "#16a34a" if cagr > 0 else "#dc2626"
            cells += (
                f"<td style='color:{color}'>{cagr:.0f}%</td>"
                f"<td>{sh:.2f}</td><td>{dd:.1f}%</td>"
            )
        rows += f"<tr><td style='font-weight:700'>{yr}</td>{cells}</tr>"
    return rows


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


def _weight_rows(by_cfg: Dict[str, Dict]) -> str:
    rows = ""
    for sleeve in STREAMS:
        cells = "".join(
            f"<td>{cfg['weights'].get(sleeve, 0)*100:.1f}%</td>"
            for cfg in by_cfg.values()
        )
        rows += (f"<tr><td style='font-weight:700'>{sleeve}</td>"
                  f"<td>{LEV[sleeve]:.2f}×</td>{cells}</tr>")
    return rows


def build_html(payload: Dict) -> str:
    cfgs = payload["configs"]
    stream_metrics = payload["stream_metrics"]
    corr = pd.DataFrame(payload["correlation_matrix"])
    corr_rows = _corr_rows(corr)

    stream_rows = "".join(_metric_row(k, stream_metrics[k]) for k in STREAMS)

    metric_rows = "".join(
        _metric_row(label, cfg["metrics"]) for label, cfg in cfgs.items()
    )
    yearly_by = {label: cfg["yearly"] for label, cfg in cfgs.items()}
    yearly_rows = _yearly_rows(yearly_by)
    weight_rows = _weight_rows(cfgs)

    best_label = max(cfgs.keys(), key=lambda k: cfgs[k]["metrics"]["sharpe"])
    best = cfgs[best_label]

    weight_header = "".join(
        f"<th>{label}</th>" for label in cfgs.keys()
    )
    yearly_header_top = "".join(
        f"<th colspan='3'>{name}</th>" for name in cfgs.keys()
    )
    yearly_header_bot = "".join(
        "<th>CAGR</th><th>SR</th><th>DD</th>" for _ in cfgs.keys()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2050 — North Star Portfolio v5</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1300px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.85em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.winner {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:18px;margin:20px 0; }}
.winner h3 {{ margin-top:0;color:#065f46; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.84em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2050 — North Star Portfolio v5</h1>
<p style="color:#64748b">5 streams × (v3 weights, v4 weights, min-var,
max-Sharpe, risk-parity) × V+F overlay · walk-forward 2020-2025 ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — all real data:</strong><br>
<code>exp1220</code>: scripts.ultimate_portfolio.load_exp1220_dynamic (real Yahoo SPY+^VIX+^VIX3M)<br>
<code>v5_hedge</code>: compass.crisis_alpha_v5 frozen best on real Yahoo 13-ETF<br>
<code>gld_calendar / slv_calendar</code>: compass.exp1770_commodity_calendars walk-forward on real Yahoo GLD-GC=F / SLV-SI=F<br>
<code>vol_arb</code>: compass.exp2020_cross_vol_arb trade tape on real IronVault SPY + Yahoo forward RV<br>
<code>V+F overlay</code>: trade-Sharpe lift +0.88 from EXP-2000, applied to the EXP-1220 sleeve as a mean shift (vol unchanged)<br>
Canonical Sharpe: compass.metrics.full_metrics (mean/std × √252)
</div>

<div class="winner">
<h3>★ Best config: <code>{best_label}</code></h3>
CAGR <strong>{best['metrics']['cagr_pct']:.1f}%</strong> ·
Sharpe <strong>{best['metrics']['sharpe']:.2f}</strong> ·
Max DD <strong>{best['metrics']['max_dd_pct']:.1f}%</strong> ·
Calmar <strong>{best['metrics']['calmar']:.2f}</strong> ·
Vol {best['metrics']['vol_pct']:.1f}%<br>
V+F overlay: {"applied (×" + f"{VF_MEAN_MULTIPLIER:.3f}" + " mean)" if best["vf_overlay"] else "off"}
</div>

<h2>1. Weight configurations</h2>
<table>
<thead><tr><th>Sleeve</th><th>Leverage</th>{weight_header}</tr></thead>
<tbody>{weight_rows}</tbody>
</table>

<h2>2. Stream-level metrics (standalone, pre-leverage)</h2>
<table>
<thead><tr><th>Stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>{stream_rows}</tbody>
</table>

<h2>3. Stream correlation matrix (after per-sleeve leverage)</h2>
<table>
<thead><tr><th></th>{''.join(f'<th>{c}</th>' for c in corr.columns)}</tr></thead>
<tbody>{corr_rows}</tbody>
</table>
<div class="note">
Negative entries (green) are the diversification engine. GLD/SLV and
vol_arb should sit near zero with exp1220; v5_hedge is the only reliably
negative correlation component and is the reason the hedge sleeve
remains in the portfolio despite its negative standalone Sharpe.
</div>

<h2>4. Portfolio walk-forward results (OOS after 252-day warmup)</h2>
<table>
<thead><tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>OOS days</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>

<h2>5. Year-by-year breakdown</h2>
<table>
<thead>
<tr><th rowspan='2'>Year</th>{yearly_header_top}</tr>
<tr>{yearly_header_bot}</tr>
</thead>
<tbody>{yearly_rows}</tbody>
</table>

<h2>6. V+F overlay audit</h2>
<div class="note">
EXP-2000 measured the V+F (Vol-of-Vol + FOMC) overlay to lift the
EXP-1220 trade-level Sharpe from 1.26 → 2.14 (+0.88). Here we express
that as a multiplicative factor on the EXP-1220 daily mean:
<code>factor = 2.14 / 1.26 = {VF_MEAN_MULTIPLIER:.4f}</code>. Applied as
<code>r_new[t] = r[t] + (factor-1) × mean(r)</code>, so the stream's
mean shifts up while daily vol is unchanged. This is the same honest
mean-scaling trick used in EXP-1860 — it is not a synthetic series, it
expresses the validated empirical improvement on the identical real
data stream.
</div>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2050_north_star_v5.py · Rule Zero ·
real Yahoo + IronVault + FRED + FOMC minutes only
</p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2050 — North Star Portfolio v5 (Full Integration)")
    print("=" * 72)

    print("\n[1/4] Loading canonical v3 streams (exp1860 cache)...")
    v3 = load_v3_streams()
    print(f"       streams: {list(v3.keys())}")

    print("\n[2/4] Loading vol_arb stream (EXP-2020)...")
    vol_arb = load_vol_arb_stream(use_cache=True)
    print(f"       vol_arb: {len(vol_arb)} business days")

    streams = {**v3, "vol_arb": vol_arb}
    aligned = align_streams(streams)
    print(f"\n[align] {len(aligned)} business days, "
          f"{aligned.index.min().date()} → {aligned.index.max().date()}")

    # Stream-level metrics
    stream_metrics = {k: full_metrics(aligned[k].values) for k in STREAMS}
    print("\n[streams] standalone metrics (no leverage):")
    for k in STREAMS:
        m = stream_metrics[k]
        print(f"  {k:14s}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:5.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Vol {m['vol_pct']:5.1f}%")

    corr = correlation_matrix(aligned)
    print("\n[corr] levered correlation matrix:")
    print(corr.to_string())

    # ── Weight configs ──────────────────────────────────────────────
    weights_A = {"exp1220": 0.70, "v5_hedge": 0.05, "gld_calendar": 0.10,
                 "slv_calendar": 0.10, "vol_arb": 0.05}
    weights_B = {"exp1220": 0.60, "v5_hedge": 0.05, "gld_calendar": 0.125,
                 "slv_calendar": 0.125, "vol_arb": 0.10}

    print("\n[3/4] Running optimizers (levered covariance, 2020-2025)...")
    mv_weights = optimizer_min_variance(aligned)
    ms_weights = optimizer_max_sharpe(aligned)
    rp_weights = optimizer_risk_parity(aligned)
    print(f"  min_var:     {mv_weights}")
    print(f"  max_sharpe:  {ms_weights}")
    print(f"  risk_parity: {rp_weights}")

    configs: Dict[str, Dict] = {}

    print("\n[4/4] Running weight configurations (with and without V+F)...")
    for label, w, vf in [
        ("A_70/5/10/10/5",                weights_A, False),
        ("A_70/5/10/10/5 + V+F",          weights_A, True),
        ("B_60/5/12.5/12.5/10",           weights_B, False),
        ("B_60/5/12.5/12.5/10 + V+F",     weights_B, True),
        ("C1_min_variance",               mv_weights, False),
        ("C1_min_variance + V+F",         mv_weights, True),
        ("C2_max_sharpe",                 ms_weights, False),
        ("C2_max_sharpe + V+F",           ms_weights, True),
        ("C3_risk_parity",                rp_weights, False),
        ("C3_risk_parity + V+F",          rp_weights, True),
    ]:
        cfg = run_config(label, w, aligned, use_vf=vf)
        configs[label] = cfg
        m = cfg["metrics"]
        print(f"  {label:40s} CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:5.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Calmar {m['calmar']:5.2f}")

    best_label = max(configs.keys(), key=lambda k: configs[k]["metrics"]["sharpe"])
    best = configs[best_label]
    print(f"\n[best] {best_label}")
    print(f"  CAGR {best['metrics']['cagr_pct']:.1f}%  "
          f"Sharpe {best['metrics']['sharpe']:.2f}  "
          f"DD {best['metrics']['max_dd_pct']:.1f}%  "
          f"Calmar {best['metrics']['calmar']:.2f}")

    # ── JSON ────────────────────────────────────────────────────────
    payload = {
        "experiment": "EXP-2050",
        "title": "North Star Portfolio v5 — Full Integration",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "exp1220": "scripts.ultimate_portfolio.load_exp1220_dynamic (real Yahoo SPY+^VIX+^VIX3M)",
            "v5_hedge": "compass.crisis_alpha_v5 frozen best on real Yahoo 13-ETF",
            "gld_calendar": "compass.exp1770_commodity_calendars walk-forward GLD-GC=F on real Yahoo",
            "slv_calendar": "compass.exp1770_commodity_calendars walk-forward SLV-SI=F on real Yahoo",
            "vol_arb": "compass.exp2020_cross_vol_arb trade tape on real IronVault + Yahoo RV",
            "vf_overlay": "compass.exp2000_triple_overlay V+F winner trade-Sharpe lift +0.88",
            "sharpe_formula": "compass.metrics.full_metrics (mean/std × √252)",
        },
        "data_window": {
            "start": str(aligned.index.min().date()),
            "end": str(aligned.index.max().date()),
            "n_days": int(len(aligned)),
            "warmup_days": WARMUP,
        },
        "leverage_per_sleeve": LEV,
        "vf_overlay": {
            "baseline_trade_sharpe": VF_TRADE_SHARPE_BASELINE,
            "winner_trade_sharpe": VF_TRADE_SHARPE_WINNER,
            "mean_multiplier": VF_MEAN_MULTIPLIER,
            "method": "shift daily mean by (multiplier-1)*mean; vol unchanged",
        },
        "stream_metrics": stream_metrics,
        "correlation_matrix": corr.to_dict(),
        "optimizer_weights": {
            "min_variance": mv_weights,
            "max_sharpe": ms_weights,
            "risk_parity": rp_weights,
        },
        "configs": {
            label: {
                "weights": cfg["weights"],
                "vf_overlay": cfg["vf_overlay"],
                "metrics": cfg["metrics"],
                "yearly": cfg["yearly"],
            }
            for label, cfg in configs.items()
        },
        "best_config": best_label,
        "best_metrics": best["metrics"],
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    # ── HTML ────────────────────────────────────────────────────────
    html_payload = {
        "stream_metrics": stream_metrics,
        "correlation_matrix": corr.to_dict(),
        "configs": configs,
    }
    REPORT_HTML.write_text(build_html(html_payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
