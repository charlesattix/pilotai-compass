"""
EXP-760-max: ML Filter (P>=0.60) + Dynamic Regime Sizing.

Reuses EXP-710 walk-forward ensemble, filters at P>=0.60, then applies
regime-dependent PnL multipliers.  Sweeps multiplier configurations.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "training_data_combined.csv"
RESULTS_DIR = ROOT / "results"

ML_THRESHOLD = 0.60  # best total-PnL threshold from EXP-710

FEATURES = [
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
    "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d",
    "realized_vol_20d",
    "net_credit", "spread_width", "max_loss_per_unit",
]

STARTING_CAPITAL = 100_000.0

# Regime multiplier configs to sweep
CONFIGS: Dict[str, Dict[str, float]] = {
    "baseline":        {"bull": 1.00, "neutral": 1.00, "bear": 1.00, "high_vol": 1.00, "crash": 1.00, "low_vol": 1.00},
    "conservative":    {"bull": 1.25, "neutral": 1.00, "bear": 0.75, "high_vol": 0.50, "crash": 0.25, "low_vol": 1.00},
    "moderate":        {"bull": 1.50, "neutral": 1.00, "bear": 0.50, "high_vol": 0.25, "crash": 0.10, "low_vol": 1.00},
    "aggressive":      {"bull": 2.00, "neutral": 1.00, "bear": 0.50, "high_vol": 0.25, "crash": 0.10, "low_vol": 1.00},
    "very_aggressive": {"bull": 2.50, "neutral": 1.00, "bear": 0.30, "high_vol": 0.10, "crash": 0.10, "low_vol": 1.00},
    "bull_only":       {"bull": 2.00, "neutral": 0.50, "bear": 0.25, "high_vol": 0.10, "crash": 0.00, "low_vol": 0.50},
    "bear_hedge":      {"bull": 1.50, "neutral": 1.00, "bear": 0.75, "high_vol": 0.50, "crash": 0.25, "low_vol": 1.00},
}


# ── Helpers ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["entry_date", "exit_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def profit_factor(pnls: np.ndarray) -> float:
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def annual_return(equity: np.ndarray, days: int) -> float:
    if len(equity) < 2 or equity[0] <= 0 or days <= 0:
        return 0.0
    total = equity[-1] / equity[0]
    years = days / 252
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


# ── Walk-forward (reuse from EXP-710) ──────────────────────────────────

def train_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].unique())
    df = df.copy()
    df["pred_prob"] = np.nan
    available_features = [f for f in FEATURES if f in df.columns]

    for i, test_year in enumerate(years):
        if i == 0:
            continue
        train_mask = df["year"].isin(years[:i])
        test_mask = df["year"] == test_year
        X_train = df.loc[train_mask, available_features].values.astype(float)
        y_train = df.loc[train_mask, "win"].values.astype(int)
        X_test = df.loc[test_mask, available_features].values.astype(float)

        if len(X_train) < 20 or len(X_test) < 5:
            continue

        scaler = StandardScaler()
        X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
        X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X_train_s, y_train)
        xgb_prob = xgb.predict_proba(X_test_s)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        rf_prob = rf.predict_proba(X_test_s)[:, 1]

        df.loc[test_mask, "pred_prob"] = 0.5 * xgb_prob + 0.5 * rf_prob

        y_test = df.loc[test_mask, "win"].values.astype(int)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, df.loc[test_mask, "pred_prob"].values)
            print(f"  Year {test_year}: AUC={auc:.3f}  n={len(X_test)}")

    return df


# ── Backtest with regime sizing ─────────────────────────────────────────

def backtest_config(
    df: pd.DataFrame,
    config_name: str,
    multipliers: Dict[str, float],
) -> Dict[str, Any]:
    """Backtest ML-filtered trades with regime-dependent sizing."""
    oos = df.dropna(subset=["pred_prob"]).copy()
    selected = oos[oos["pred_prob"] >= ML_THRESHOLD].copy()

    if len(selected) == 0:
        return {"config": config_name, "n_trades": 0}

    # Apply regime multipliers
    selected["regime_mult"] = selected["regime"].map(multipliers).fillna(1.0)
    selected["sized_pnl"] = selected["pnl"] * selected["regime_mult"]

    pnls = selected["sized_pnl"].values
    raw_pnls = selected["pnl"].values
    wins_raw = selected["win"].values
    dates = pd.to_datetime(selected["entry_date"])

    # Equity
    equity = STARTING_CAPITAL + np.cumsum(pnls)
    date_range = max((dates.max() - dates.min()).days, 1)

    # Win rate on sized PnL (a sized-down loss is still a loss)
    wins_sized = (pnls > 0).astype(int)

    # Regime breakdown
    regime_stats = {}
    for regime in sorted(selected["regime"].unique()):
        mask = selected["regime"] == regime
        r_pnls = pnls[mask.values]
        regime_stats[regime] = {
            "n": int(mask.sum()),
            "mult": multipliers.get(regime, 1.0),
            "pnl": float(r_pnls.sum()),
            "wr": float((r_pnls > 0).mean()) if len(r_pnls) > 0 else 0,
            "avg_pnl": float(r_pnls.mean()) if len(r_pnls) > 0 else 0,
        }

    daily_returns = pnls / STARTING_CAPITAL

    return {
        "config": config_name,
        "multipliers": multipliers,
        "n_trades": len(selected),
        "win_rate_raw": float(wins_raw.mean()),
        "win_rate_sized": float(wins_sized.mean()),
        "total_pnl": float(pnls.sum()),
        "total_pnl_raw": float(raw_pnls.sum()),
        "pnl_improvement": float(pnls.sum() - raw_pnls.sum()),
        "annual_return": annual_return(equity, date_range),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(daily_returns),
        "profit_factor": min(profit_factor(pnls), 99.9),
        "avg_pnl": float(pnls.mean()),
        "regime_breakdown": regime_stats,
    }


# ── HTML report ─────────────────────────────────────────────────────────

def generate_html(
    results: List[Dict],
    optimal: Dict,
    baseline: Dict,
    exp710_ref: Dict,
    df: pd.DataFrame,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    opt_name = optimal["config"]

    # Config comparison table
    rows = ""
    for r in results:
        cls = ' style="background:#f0fdf4;font-weight:600"' if r["config"] == opt_name else ""
        rows += (
            f'<tr{cls}>'
            f'<td style="text-align:left">{r["config"]}</td>'
            f'<td>{r["n_trades"]}</td>'
            f'<td>{r["win_rate_sized"]:.1%}</td>'
            f'<td>${r["total_pnl"]:+,.0f}</td>'
            f'<td>{r["annual_return"]:+.1%}</td>'
            f'<td>{r["max_drawdown"]:.1%}</td>'
            f'<td>{r["sharpe"]:.2f}</td>'
            f'<td>{r["profit_factor"]:.2f}</td>'
            f'<td>${r["pnl_improvement"]:+,.0f}</td>'
            f'</tr>\n'
        )

    # Regime breakdown for optimal
    regime_rows = ""
    for regime, stats in sorted(optimal.get("regime_breakdown", {}).items()):
        regime_rows += (
            f'<tr><td style="text-align:left">{regime}</td>'
            f'<td>{stats["mult"]:.2f}x</td>'
            f'<td>{stats["n"]}</td>'
            f'<td>{stats["wr"]:.0%}</td>'
            f'<td>${stats["pnl"]:+,.0f}</td>'
            f'<td>${stats["avg_pnl"]:+,.0f}</td></tr>\n'
        )

    # SVG bar chart: PnL by config
    chart = _svg_config_comparison(results, opt_name)
    regime_chart = _svg_regime_pnl(optimal)

    # Comparison vs EXP-710
    e710_sharpe = exp710_ref.get("sharpe", 0)
    e710_pnl = exp710_ref.get("total_pnl", 0)
    e710_dd = exp710_ref.get("max_drawdown", 0)
    e710_wr = exp710_ref.get("win_rate", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EXP-760-max: ML Filter + Regime Sizing</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }}
  h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px;
          padding:1em 1.5em; min-width:130px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }}
  .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left;
       border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px;
            padding:1em; margin:1.5em 0; text-align:center; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.8em; color:#94a3b8; }}
</style>
</head>
<body>

<h1>EXP-760-max: ML Filter + Dynamic Regime Sizing</h1>
<div class="meta">ML threshold P≥{ML_THRESHOLD} &middot; {len(results)} configs tested &middot; {optimal['n_trades']} trades &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value good">{opt_name}</div><div class="label">Optimal Config</div></div>
  <div class="kpi"><div class="value">{optimal['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value good">{optimal['win_rate_sized']:.0%}</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">{optimal['max_drawdown']:.1%}</div><div class="label">Max DD</div></div>
  <div class="kpi"><div class="value">${optimal['total_pnl']:+,.0f}</div><div class="label">Total P&L</div></div>
  <div class="kpi"><div class="value good">${optimal['pnl_improvement']:+,.0f}</div><div class="label">vs No Sizing</div></div>
</div>

<h2>1. Configuration Comparison</h2>
<div class="chart">{chart}</div>
<table>
<thead><tr><th>Config</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th>
<th>Ann Return</th><th>Max DD</th><th>Sharpe</th><th>PF</th><th>P&L Improvement</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>2. Optimal Config Regime Breakdown</h2>
<div class="chart">{regime_chart}</div>
<table>
<thead><tr><th>Regime</th><th>Multiplier</th><th>Trades</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th></tr></thead>
<tbody>{regime_rows}</tbody>
</table>

<h2>3. vs EXP-710 (ML Filter Only, P≥0.60)</h2>
<table>
<thead><tr><th>Metric</th><th>EXP-710 (P≥0.60)</th><th>EXP-760 ({opt_name})</th><th>Change</th></tr></thead>
<tbody>
<tr><td style="text-align:left">Win Rate</td><td>{e710_wr:.1%}</td><td>{optimal['win_rate_sized']:.1%}</td><td>{(optimal['win_rate_sized']-e710_wr)*100:+.1f}pp</td></tr>
<tr><td style="text-align:left">Sharpe</td><td>{e710_sharpe:.2f}</td><td>{optimal['sharpe']:.2f}</td><td class="{"good" if optimal['sharpe']>e710_sharpe else "bad"}">{optimal['sharpe']-e710_sharpe:+.2f}</td></tr>
<tr><td style="text-align:left">Max DD</td><td>{e710_dd:.1%}</td><td>{optimal['max_drawdown']:.1%}</td><td>{(optimal['max_drawdown']-e710_dd)*100:+.1f}pp</td></tr>
<tr><td style="text-align:left">Total P&L</td><td>${e710_pnl:+,.0f}</td><td>${optimal['total_pnl']:+,.0f}</td><td class="{"good" if optimal['total_pnl']>e710_pnl else "bad"}">${optimal['total_pnl']-e710_pnl:+,.0f}</td></tr>
</tbody>
</table>

<footer>Generated by <code>EXP-760-max/backtest.py</code></footer>
</body></html>"""
    return html


def _svg_config_comparison(results: List[Dict], optimal: str) -> str:
    w, h = 600, 260
    ml, mr, mt, mb = 110, 20, 25, 20
    pw = w - ml - mr
    ph = h - mt - mb

    pnls = [r["total_pnl"] for r in results]
    if not pnls:
        return ""
    max_pnl = max(abs(p) for p in pnls) or 1
    bar_h = ph / len(results) * 0.7
    gap = ph / len(results)

    elements = ""
    for i, r in enumerate(results):
        y = mt + i * gap
        bw = abs(r["total_pnl"]) / max_pnl * pw
        x = ml
        color = "#16a34a" if r["total_pnl"] > 0 else "#dc2626"
        if r["config"] == optimal:
            color = "#0f766e"
        elements += (
            f'<rect x="{x}" y="{y:.0f}" width="{bw:.0f}" height="{bar_h:.0f}" '
            f'fill="{color}" opacity="0.8" rx="2"/>'
            f'<text x="{ml-5}" y="{y+bar_h*0.7:.0f}" text-anchor="end" '
            f'font-size="10" fill="#334155">{r["config"]}</text>'
            f'<text x="{x+bw+5:.0f}" y="{y+bar_h*0.7:.0f}" '
            f'font-size="9" fill="{color}">${r["total_pnl"]:+,.0f} (S={r["sharpe"]:.1f})</text>'
        )

    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{elements}</svg>'


def _svg_regime_pnl(optimal: Dict) -> str:
    rd = optimal.get("regime_breakdown", {})
    if not rd:
        return ""
    w, h = 500, 200
    ml, mr, mt, mb = 80, 20, 25, 20
    pw = w - ml - mr
    ph = h - mt - mb

    regimes = sorted(rd.keys())
    pnls = [rd[r]["pnl"] for r in regimes]
    max_abs = max(abs(p) for p in pnls) or 1
    bar_h = ph / len(regimes) * 0.7
    gap = ph / len(regimes)

    colors = {"bull": "#16a34a", "bear": "#dc2626", "high_vol": "#f59e0b",
              "crash": "#7f1d1d", "neutral": "#64748b", "low_vol": "#3b82f6"}

    elements = ""
    mid_x = ml + pw / 2
    elements += f'<line x1="{mid_x}" y1="{mt}" x2="{mid_x}" y2="{h-mb}" stroke="#cbd5e1" stroke-dasharray="3"/>'

    for i, regime in enumerate(regimes):
        y = mt + i * gap
        pnl = rd[regime]["pnl"]
        bw = abs(pnl) / max_abs * (pw / 2)
        if pnl >= 0:
            x = mid_x
        else:
            x = mid_x - bw
        c = colors.get(regime, "#64748b")
        elements += (
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{bar_h:.0f}" '
            f'fill="{c}" opacity="0.8" rx="2"/>'
            f'<text x="{ml-5}" y="{y+bar_h*0.7:.0f}" text-anchor="end" '
            f'font-size="10" fill="#334155">{regime} ({rd[regime]["mult"]}x)</text>'
        )

    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{elements}</svg>'


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXP-760-max: ML Filter + Dynamic Regime Sizing")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  {len(df)} trades, regimes: {df['regime'].value_counts().to_dict()}")

    print("\n[2/5] Walk-forward ensemble training...")
    df = train_walk_forward(df)
    oos = df.dropna(subset=["pred_prob"])
    ml_filtered = oos[oos["pred_prob"] >= ML_THRESHOLD]
    print(f"\n  ML filter (P>={ML_THRESHOLD}): {len(ml_filtered)} of {len(oos)} OOS trades")
    print(f"  Regime distribution in filtered: {ml_filtered['regime'].value_counts().to_dict()}")

    print("\n[3/5] Sweeping regime multiplier configurations...")
    results = []
    for config_name, mults in CONFIGS.items():
        r = backtest_config(df, config_name, mults)
        results.append(r)
        print(f"  {config_name:20s}: P&L=${r['total_pnl']:+10,.0f}  "
              f"Sharpe={r['sharpe']:6.2f}  DD={r['max_drawdown']:7.1%}  "
              f"WR={r['win_rate_sized']:.0%}  Δ=${r['pnl_improvement']:+,.0f}")

    # EXP-710 reference (baseline = no sizing)
    exp710_ref = {
        "sharpe": results[0]["sharpe"],
        "total_pnl": results[0]["total_pnl"],
        "max_drawdown": results[0]["max_drawdown"],
        "win_rate": results[0]["win_rate_raw"],
    }

    print("\n[4/5] Finding optimal configuration...")
    # Optimal: best Sharpe
    optimal = max(results, key=lambda r: r["sharpe"])
    print(f"\n  >> Optimal: {optimal['config']}")
    print(f"     Sharpe={optimal['sharpe']:.2f}, P&L=${optimal['total_pnl']:+,.0f}, "
          f"DD={optimal['max_drawdown']:.1%}, WR={optimal['win_rate_sized']:.0%}")
    print(f"     P&L improvement over no sizing: ${optimal['pnl_improvement']:+,.0f}")

    print("\n[5/5] Generating outputs...")
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "experiment": "EXP-760-max",
        "description": "ML Filter + Dynamic Regime Sizing",
        "generated": datetime.now().isoformat(),
        "ml_threshold": ML_THRESHOLD,
        "data": {
            "total_trades": len(df),
            "oos_trades": len(oos),
            "ml_filtered_trades": len(ml_filtered),
        },
        "optimal": {k: v for k, v in optimal.items() if k != "regime_breakdown"},
        "optimal_regime_breakdown": optimal.get("regime_breakdown", {}),
        "exp710_reference": exp710_ref,
        "all_configs": [{k: v for k, v in r.items() if k != "regime_breakdown"} for r in results],
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Wrote results/summary.json")

    html = generate_html(results, optimal, results[0], exp710_ref, df)
    (RESULTS_DIR / "report.html").write_text(html)
    print("  Wrote results/report.html")

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
