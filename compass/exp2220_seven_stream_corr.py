"""
EXP-2220 — Full Pairwise Correlation Matrix on the 7-Stream Portfolio
======================================================================

Streams (all real, all Rule-Zero clean):
  1. exp1220     SPY put-credit-spreads     (compass.exp2080 cube)
  2. v5_hedge    Crisis Alpha v5            (compass.exp2080 cube)
  3. gld_cal     GLD calendar               (compass.exp2080 cube)
  4. slv_cal     SLV calendar               (compass.exp2080 cube)
  5. cross_vol   cross-sectional vol arb    (compass.exp2080 cube)
  6. xlf_cs      XLF put-credit-spreads     (compass.exp2160 engine)
  7. xli_cs      XLI put-credit-spreads     (compass.exp2160 engine)

Computes
--------
* Static pairwise correlation matrix (Pearson + Spearman)
* Rolling 60-day Pearson for every pair → mean / min / max stability
* Drawdown-conditional correlation: Pearson restricted to days when
  EXP-1220's equity curve is in drawdown ≥ −1% from its rolling peak
  (the regime that matters for portfolio survival)
* Eigenvalue decomposition of the correlation matrix → effective
  number of independent bets, dominant principal-component loadings

Outputs
  compass/reports/exp2220_seven_stream_corr.json
  compass/reports/exp2220_seven_stream_corr.html

Honest scope note: this is a *measurement* experiment. It does not
build a portfolio, optimise weights, or claim alpha. Its job is to
hand the next stage (EXP-2230 or production) a correct picture of how
much of the 7 streams' apparent diversification is real.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2220_seven_stream_corr.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2220_seven_stream_corr.html"

ROLL_WINDOW = 60
DD_REFERENCE = "exp1220"   # the reference stream for drawdown filtering
DD_THRESHOLD = -0.01       # -1% from rolling peak


# ─────────────────────────────────────────────────────────────────────────────
# Stream loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_xlf_xli_streams(index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """Build XLF and XLI credit-spread daily series via EXP-2160 engine."""
    from compass.exp2160_high_capacity_alts import (
        run_put_credit_spreads, trades_to_daily_pct,
    )
    out: Dict[str, pd.Series] = {}
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    try:
        for tk in ("XLF", "XLI"):
            trades = run_put_credit_spreads(con, tk)
            daily = trades_to_daily_pct(trades, index)
            out[f"{tk.lower()}_cs"] = daily.rename(f"{tk.lower()}_cs")
            print(f"      {tk}: {len(trades)} trades, "
                  f"sum daily pct = {float(daily.sum()):.4f}")
    finally:
        con.close()
    return out


def build_seven_stream_cube() -> pd.DataFrame:
    print("[1/4] loading 5-stream cached cube …")
    base = load_streams()
    print(f"      {base.shape}, range {base.index[0].date()} → {base.index[-1].date()}")

    print("[2/4] building XLF + XLI credit-spread streams (real IronVault) …")
    extras = load_xlf_xli_streams(base.index)
    df = base.copy()
    for k, s in extras.items():
        df[k] = s.reindex(df.index).fillna(0.0)
    df = df[["exp1220", "v5_hedge", "gld_cal", "slv_cal", "cross_vol", "xlf_cs", "xli_cs"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
def static_correlations(df: pd.DataFrame) -> Dict:
    pearson  = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    cols = list(df.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({
                "a": a, "b": b,
                "pearson":  round(float(pearson.loc[a, b]),  4),
                "spearman": round(float(spearman.loc[a, b]), 4),
            })
    pairs.sort(key=lambda r: abs(r["pearson"]), reverse=True)
    return {
        "pearson":  {a: {b: round(float(pearson.loc[a, b]),  4)  for b in cols} for a in cols},
        "spearman": {a: {b: round(float(spearman.loc[a, b]), 4) for b in cols} for a in cols},
        "pairs_sorted_by_abs_pearson": pairs,
    }


def rolling_correlation_stats(df: pd.DataFrame, window: int = ROLL_WINDOW) -> Dict:
    cols = list(df.columns)
    out: Dict[str, Dict] = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            roll = df[a].rolling(window).corr(df[b]).dropna()
            if len(roll) < 10:
                continue
            out[f"{a} vs {b}"] = {
                "n_obs": int(len(roll)),
                "mean":  round(float(roll.mean()),  4),
                "std":   round(float(roll.std(ddof=1)),   4),
                "min":   round(float(roll.min()),   4),
                "max":   round(float(roll.max()),   4),
                "p10":   round(float(roll.quantile(0.10)), 4),
                "p90":   round(float(roll.quantile(0.90)), 4),
            }
    return out


def drawdown_mask(series: pd.Series, threshold: float = DD_THRESHOLD) -> pd.Series:
    eq = (1 + series).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return dd <= threshold


def drawdown_conditional_correlation(df: pd.DataFrame,
                                     reference: str = DD_REFERENCE,
                                     threshold: float = DD_THRESHOLD) -> Dict:
    if reference not in df.columns:
        return {}
    mask = drawdown_mask(df[reference], threshold)
    inside = df.loc[mask]
    outside = df.loc[~mask]
    cols = list(df.columns)

    def _matrix(sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if len(sub) < 5:
            return {a: {b: None for b in cols} for a in cols}
        c = sub.corr(method="pearson")
        return {a: {b: round(float(c.loc[a, b]), 4) for b in cols} for a in cols}

    inside_pairs = []
    if len(inside) >= 5:
        ci = inside.corr(method="pearson")
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                inside_pairs.append({
                    "a": a, "b": b,
                    "corr_inside":  round(float(ci.loc[a, b]), 4),
                })
        inside_pairs.sort(key=lambda r: abs(r["corr_inside"]), reverse=True)

    return {
        "reference_stream": reference,
        "dd_threshold": threshold,
        "n_days_in_dd":  int(mask.sum()),
        "n_days_normal": int((~mask).sum()),
        "pct_in_dd":     round(float(mask.mean()) * 100, 2),
        "matrix_in_dd":   _matrix(inside),
        "matrix_normal":  _matrix(outside),
        "pairs_in_dd_sorted": inside_pairs,
    }


def eigen_decomposition(df: pd.DataFrame) -> Dict:
    """PCA-style eigen analysis of the correlation matrix.

    Effective number of independent streams ~ exp(entropy of normalised
    eigenvalues), or equivalently 1 / Σ p_i² (the participation ratio).
    """
    C = df.corr(method="pearson").values
    cols = list(df.columns)
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh sorts ascending → flip to descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    p = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
    # Effective # of independent streams
    eff_n_pr = float(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 0 else 0.0
    eff_n_entropy = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))

    components = []
    for k in range(len(cols)):
        loadings = {cols[i]: round(float(eigvecs[i, k]), 4) for i in range(len(cols))}
        components.append({
            "k": k + 1,
            "eigenvalue": round(float(eigvals[k]), 4),
            "explained_pct": round(float(p[k]) * 100, 2),
            "cumulative_pct": round(float(p[: k + 1].sum()) * 100, 2),
            "loadings": loadings,
        })
    return {
        "eigenvalues": [round(float(v), 4) for v in eigvals],
        "explained_pct": [round(float(p_i) * 100, 2) for p_i in p],
        "cumulative_pct": [round(float(p[: k + 1].sum()) * 100, 2) for k in range(len(cols))],
        "effective_n_streams_participation": round(eff_n_pr, 3),
        "effective_n_streams_entropy":      round(eff_n_entropy, 3),
        "components": components,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = build_seven_stream_cube()
    print(f"      cube shape {df.shape}  cols {list(df.columns)}")

    print("[3/4] computing correlation analyses …")
    static = static_correlations(df)
    rolling = rolling_correlation_stats(df, ROLL_WINDOW)
    dd_cond = drawdown_conditional_correlation(df)
    eig = eigen_decomposition(df)

    print("[4/4] writing report …")
    summary = {
        "n_streams": df.shape[1],
        "n_days": int(len(df)),
        "range": [str(df.index[0].date()), str(df.index[-1].date())],
        "highest_abs_corr_pair": static["pairs_sorted_by_abs_pearson"][0]
            if static["pairs_sorted_by_abs_pearson"] else None,
        "median_pair_abs_corr": round(
            float(np.median([abs(p["pearson"]) for p in static["pairs_sorted_by_abs_pearson"]])), 4
        ) if static["pairs_sorted_by_abs_pearson"] else None,
        "effective_n_streams_pr": eig["effective_n_streams_participation"],
        "effective_n_streams_entropy": eig["effective_n_streams_entropy"],
        "dd_pct_of_sample": dd_cond.get("pct_in_dd"),
    }
    payload = {
        "experiment": "EXP-2220",
        "name": "Full pairwise correlation matrix — 7-stream portfolio",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "five_streams": "compass.exp2080_corr_regime.load_streams (cached real cube)",
            "xlf_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads('XLF')",
            "xli_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads('XLI')",
        },
        "streams": list(df.columns),
        "summary": summary,
        "static_correlation": static,
        "rolling_60d_stats": rolling,
        "drawdown_conditional": dd_cond,
        "eigen_decomposition": eig,
        "honest_scope": (
            "Measurement experiment only — does NOT build a portfolio. "
            "All correlation matrices are computed on real daily-return "
            "series with no synthetic fills. Drawdown-conditional analysis "
            "uses EXP-1220 as the reference because it carries the largest "
            "static weight in the production portfolio (40%)."
        ),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _matrix_table(mat: Dict, cols: List[str]) -> str:
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    rows = ""
    for a in cols:
        cells = ""
        for b in cols:
            v = mat.get(a, {}).get(b)
            if v is None:
                cells += "<td>—</td>"; continue
            color = "#fff"
            if a == b:
                color = "#eee"
            elif v >= 0.5:
                color = "#fdd"
            elif v >= 0.2:
                color = "#fee8d8"
            elif v <= -0.5:
                color = "#cce5ff"
            elif v <= -0.2:
                color = "#dde8f5"
            cells += f"<td style='background:{color}'>{v:+.2f}</td>"
        rows += f"<tr><th>{a}</th>{cells}</tr>"
    return f"<table>{head}{rows}</table>"


def _write_html(p: Dict) -> None:
    cols = p["streams"]
    s = p["summary"]
    pearson_html  = _matrix_table(p["static_correlation"]["pearson"],  cols)
    spearman_html = _matrix_table(p["static_correlation"]["spearman"], cols)
    in_dd_html    = _matrix_table(p["drawdown_conditional"].get("matrix_in_dd",  {}), cols)
    normal_html   = _matrix_table(p["drawdown_conditional"].get("matrix_normal", {}), cols)

    rows_pairs = "".join(
        f"<tr><td>{r['a']}</td><td>{r['b']}</td>"
        f"<td>{r['pearson']:+.3f}</td><td>{r['spearman']:+.3f}</td></tr>"
        for r in p["static_correlation"]["pairs_sorted_by_abs_pearson"]
    )
    rows_roll = "".join(
        f"<tr><td>{k}</td><td>{v['mean']:+.3f}</td><td>{v['std']:.3f}</td>"
        f"<td>{v['min']:+.3f}</td><td>{v['max']:+.3f}</td>"
        f"<td>{v['p10']:+.3f}</td><td>{v['p90']:+.3f}</td></tr>"
        for k, v in p["rolling_60d_stats"].items()
    )
    rows_eig = "".join(
        f"<tr><td>PC{c['k']}</td><td>{c['eigenvalue']:.3f}</td>"
        f"<td>{c['explained_pct']:.1f}%</td><td>{c['cumulative_pct']:.1f}%</td>"
        f"<td>{', '.join(f'{k} {v:+.2f}' for k,v in sorted(c['loadings'].items(), key=lambda kv:-abs(kv[1]))[:3])}</td></tr>"
        for c in p["eigen_decomposition"]["components"]
    )

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2220 — 7-Stream Correlation Matrix</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2220 — Full Correlation Matrix, 7-Stream Portfolio</h1>
<p class='small'>Generated {p['generated']} · {s['n_streams']} streams · {s['n_days']} days
 ({p['summary']['range'][0]} → {p['summary']['range'][1]}) · Rule Zero clean.</p>

<h2>Summary</h2>
<ul>
<li>Highest |corr| pair: <b>{s['highest_abs_corr_pair']['a']} ↔ {s['highest_abs_corr_pair']['b']}</b>
    pearson {s['highest_abs_corr_pair']['pearson']:+.3f}</li>
<li>Median pairwise |corr|: <b>{s['median_pair_abs_corr']}</b></li>
<li>Effective number of independent streams:
    <b>{s['effective_n_streams_pr']}</b> (participation ratio) ·
    <b>{s['effective_n_streams_entropy']}</b> (entropy)</li>
<li>Sample fraction in EXP-1220 drawdown (≤ -1% from peak): <b>{s['dd_pct_of_sample']}%</b></li>
</ul>

<h2>Static Pearson</h2>
{pearson_html}

<h2>Static Spearman</h2>
{spearman_html}

<h2>Pearson during EXP-1220 drawdown (the regime that matters)</h2>
{in_dd_html}

<h2>Pearson outside EXP-1220 drawdown (calm periods)</h2>
{normal_html}

<h2>All pairs ranked by |Pearson|</h2>
<table><tr><th>A</th><th>B</th><th>Pearson</th><th>Spearman</th></tr>{rows_pairs}</table>

<h2>Rolling-60d correlation stability</h2>
<table>
<tr><th>Pair</th><th>mean</th><th>std</th><th>min</th><th>max</th><th>p10</th><th>p90</th></tr>
{rows_roll}
</table>

<h2>Eigen decomposition (PCA on the correlation matrix)</h2>
<table>
<tr><th>PC</th><th>λ</th><th>Explained</th><th>Cumulative</th><th>Top loadings</th></tr>
{rows_eig}
</table>

<h2>Honest scope</h2>
<p>{p['honest_scope']}</p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
