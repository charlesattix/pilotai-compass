"""
EXP-710-max: Aggressive ML Signal Filtering backtest.

Walk-forward XGBoost + RandomForest ensemble.  Train on years 1..N,
predict year N+1.  Sweep P(win) thresholds 0.50-0.90.  Output summary
JSON, HTML report, and analysis markdown.
"""

from __future__ import annotations

import json
import math
import sys
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

# ── Paths ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "training_data_combined.csv"
RESULTS_DIR = ROOT / "results"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

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


# ── Helpers ─────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["entry_date", "exit_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)
    # Fill NaN in features
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def sharpe(returns: np.ndarray, ann: float = 252.0) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(ann))


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


# ── Walk-forward ensemble ───────────────────────────────────────────────


def train_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward: train on years 1..N, predict year N+1.

    Returns DataFrame with an added 'pred_prob' column (OOS predictions).
    """
    years = sorted(df["year"].unique())
    df = df.copy()
    df["pred_prob"] = np.nan

    available_features = [f for f in FEATURES if f in df.columns]

    aucs = []

    for i, test_year in enumerate(years):
        if i == 0:
            # No training data for first year — skip (or use prior if available)
            continue

        train_years = years[:i]
        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year

        X_train = df.loc[train_mask, available_features].values.astype(float)
        y_train = df.loc[train_mask, "win"].values.astype(int)
        X_test = df.loc[test_mask, available_features].values.astype(float)
        y_test = df.loc[test_mask, "win"].values.astype(int)

        if len(X_train) < 20 or len(X_test) < 5:
            continue

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Handle NaN from scaling
        X_train_s = np.nan_to_num(X_train_s, nan=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0)

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X_train_s, y_train)
        xgb_prob = xgb.predict_proba(X_test_s)[:, 1]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        rf_prob = rf.predict_proba(X_test_s)[:, 1]

        # Ensemble: average
        ensemble_prob = 0.5 * xgb_prob + 0.5 * rf_prob

        df.loc[test_mask, "pred_prob"] = ensemble_prob

        # AUC
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, ensemble_prob)
            aucs.append((test_year, auc, len(y_test)))
            print(f"  Year {test_year}: AUC={auc:.3f}  n={len(y_test)}  "
                  f"train={len(X_train)}  WR={y_test.mean():.1%}")

    if aucs:
        avg_auc = np.mean([a for _, a, _ in aucs])
        print(f"\n  Walk-forward avg AUC: {avg_auc:.3f}")

    return df


# ── Backtest at threshold ───────────────────────────────────────────────


def backtest_threshold(
    df: pd.DataFrame, threshold: float,
) -> Dict[str, Any]:
    """Backtest: only take trades where pred_prob >= threshold."""
    # Filter to rows with OOS predictions
    oos = df.dropna(subset=["pred_prob"]).copy()
    selected = oos[oos["pred_prob"] >= threshold].copy()

    n_total = len(oos)
    n_selected = len(selected)
    n_years_covered = selected["year"].nunique() if n_selected > 0 else 0

    if n_selected == 0:
        return {
            "threshold": threshold,
            "n_trades": 0,
            "n_total_oos": n_total,
            "selectivity": 0.0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": 0.0,
            "trades_per_year": 0.0,
            "years_covered": 0,
        }

    pnls = selected["pnl"].values
    wins = selected["win"].values
    dates = pd.to_datetime(selected["entry_date"])

    # Equity curve: cumulative PnL
    cum_pnl = np.cumsum(pnls)
    equity = STARTING_CAPITAL + cum_pnl

    # Time span
    date_range = (dates.max() - dates.min()).days
    if date_range < 1:
        date_range = 252  # fallback

    # Daily returns proxy: spread PnL across hold days
    daily_returns = pnls / STARTING_CAPITAL

    wr = float(wins.mean())
    pf = profit_factor(pnls)
    sh = sharpe(daily_returns)
    mdd = max_drawdown(equity)
    ann_ret = annual_return(equity, date_range)
    trades_per_year = n_selected / max(n_years_covered, 1)

    return {
        "threshold": threshold,
        "n_trades": n_selected,
        "n_total_oos": n_total,
        "selectivity": n_selected / n_total if n_total > 0 else 0,
        "win_rate": wr,
        "total_pnl": float(pnls.sum()),
        "annual_return": ann_ret,
        "max_drawdown": mdd,
        "sharpe": sh,
        "profit_factor": min(pf, 99.9),
        "avg_pnl": float(pnls.mean()),
        "trades_per_year": trades_per_year,
        "years_covered": n_years_covered,
    }


# ── HTML report ─────────────────────────────────────────────────────────


def generate_html(results: List[Dict], optimal: Dict, df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    baseline = results[0]  # threshold 0.50 = ~all trades

    # KPIs
    opt_t = optimal["threshold"]

    # Results table rows
    rows = ""
    for r in results:
        cls = ' style="background:#f0fdf4;font-weight:600"' if r["threshold"] == opt_t else ""
        dd_str = f'{r["max_drawdown"]:.1%}'
        rows += (
            f'<tr{cls}>'
            f'<td>{r["threshold"]:.2f}</td>'
            f'<td>{r["n_trades"]}</td>'
            f'<td>{r["selectivity"]:.0%}</td>'
            f'<td>{r["win_rate"]:.1%}</td>'
            f'<td>${r["total_pnl"]:,.0f}</td>'
            f'<td>{r["annual_return"]:+.1%}</td>'
            f'<td>{dd_str}</td>'
            f'<td>{r["sharpe"]:.2f}</td>'
            f'<td>{r["profit_factor"]:.2f}</td>'
            f'<td>{r["avg_pnl"]:+,.0f}</td>'
            f'<td>{r["trades_per_year"]:.0f}</td>'
            f'</tr>\n'
        )

    # SVG: threshold vs Sharpe + win rate
    chart = _svg_threshold_chart(results)

    # SVG: equity curves
    equity_chart = _svg_equity_curves(results, df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EXP-710-max: ML Signal Filtering Backtest</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
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
  td:first-child {{ text-align:center; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px;
            padding:1em; margin:1.5em 0; text-align:center; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.8em; color:#94a3b8; }}
</style>
</head>
<body>

<h1>EXP-710-max: Aggressive ML Signal Filtering</h1>
<div class="meta">Walk-forward XGBoost + RF ensemble &middot; {len(df)} total trades &middot; Thresholds {THRESHOLDS[0]}-{THRESHOLDS[-1]} &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{opt_t:.2f}</div><div class="label">Optimal Threshold</div></div>
  <div class="kpi"><div class="value good">{optimal["win_rate"]:.0%}</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">{optimal["sharpe"]:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value">{optimal["max_drawdown"]:.1%}</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value">{optimal["n_trades"]}</div><div class="label">Trades</div></div>
  <div class="kpi"><div class="value">${optimal["total_pnl"]:+,.0f}</div><div class="label">Total P&L</div></div>
</div>

<h2>1. Threshold vs Performance</h2>
<div class="chart">{chart}</div>

<h2>2. Equity Curves by Threshold</h2>
<div class="chart">{equity_chart}</div>

<h2>3. Full Results Table</h2>
<table>
<thead><tr>
  <th>Threshold</th><th>Trades</th><th>Selectivity</th><th>Win Rate</th>
  <th>Total P&L</th><th>Ann Return</th><th>Max DD</th><th>Sharpe</th>
  <th>Profit Factor</th><th>Avg P&L</th><th>Trades/Yr</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>4. Baseline (No Filter) vs Optimal</h2>
<table>
<thead><tr><th>Metric</th><th>Baseline (≥0.50)</th><th>Optimal (≥{opt_t:.2f})</th><th>Improvement</th></tr></thead>
<tbody>
<tr><td style="text-align:left">Win Rate</td><td>{baseline["win_rate"]:.1%}</td><td>{optimal["win_rate"]:.1%}</td><td class="good">+{(optimal["win_rate"]-baseline["win_rate"])*100:.1f}pp</td></tr>
<tr><td style="text-align:left">Sharpe</td><td>{baseline["sharpe"]:.2f}</td><td>{optimal["sharpe"]:.2f}</td><td>{optimal["sharpe"]-baseline["sharpe"]:+.2f}</td></tr>
<tr><td style="text-align:left">Max Drawdown</td><td>{baseline["max_drawdown"]:.1%}</td><td>{optimal["max_drawdown"]:.1%}</td><td>{(baseline["max_drawdown"]-optimal["max_drawdown"])*100:+.1f}pp better</td></tr>
<tr><td style="text-align:left">Total P&L</td><td>${baseline["total_pnl"]:+,.0f}</td><td>${optimal["total_pnl"]:+,.0f}</td><td>${optimal["total_pnl"]-baseline["total_pnl"]:+,.0f}</td></tr>
<tr><td style="text-align:left">Profit Factor</td><td>{baseline["profit_factor"]:.2f}</td><td>{optimal["profit_factor"]:.2f}</td><td>{optimal["profit_factor"]-baseline["profit_factor"]:+.2f}</td></tr>
<tr><td style="text-align:left">Trades</td><td>{baseline["n_trades"]}</td><td>{optimal["n_trades"]}</td><td>{optimal["n_trades"]-baseline["n_trades"]}</td></tr>
</tbody>
</table>

<footer>Generated by <code>EXP-710-max/backtest.py</code></footer>
</body></html>"""
    return html


def _svg_threshold_chart(results: List[Dict]) -> str:
    """SVG dual-axis: threshold vs Sharpe (bars) + win rate (line)."""
    w, h = 600, 280
    margin_l, margin_r, margin_t, margin_b = 50, 50, 30, 40
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b

    n = len(results)
    if n == 0:
        return ""
    bar_w = pw / n * 0.6
    gap = pw / n

    sharpes = [r["sharpe"] for r in results]
    wrs = [r["win_rate"] for r in results]
    max_sh = max(max(abs(s) for s in sharpes), 0.1)

    elements = ""
    # Axes
    elements += f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{h-margin_b}" stroke="#cbd5e1"/>'
    elements += f'<line x1="{margin_l}" y1="{h-margin_b}" x2="{w-margin_r}" y2="{h-margin_b}" stroke="#cbd5e1"/>'

    zero_y = margin_t + ph * (max_sh / (2 * max_sh))
    elements += f'<line x1="{margin_l}" y1="{zero_y:.0f}" x2="{w-margin_r}" y2="{zero_y:.0f}" stroke="#e2e8f0" stroke-dasharray="4"/>'

    for i, r in enumerate(results):
        x = margin_l + i * gap + gap * 0.2
        # Sharpe bar
        sh = r["sharpe"]
        bar_h = abs(sh) / (2 * max_sh) * ph
        if sh >= 0:
            by = zero_y - bar_h
        else:
            by = zero_y
        color = "#16a34a" if sh > 0 else "#dc2626"
        elements += f'<rect x="{x:.0f}" y="{by:.0f}" width="{bar_w:.0f}" height="{bar_h:.0f}" fill="{color}" opacity="0.7" rx="2"/>'
        elements += f'<text x="{x + bar_w/2:.0f}" y="{h-margin_b+15}" text-anchor="middle" font-size="10" fill="#334155">{r["threshold"]:.2f}</text>'
        elements += f'<text x="{x + bar_w/2:.0f}" y="{by-3:.0f}" text-anchor="middle" font-size="9" fill="{color}">{sh:.2f}</text>'

        # Win rate dot
        wr_y = margin_t + (1 - r["win_rate"]) * ph
        cx = x + bar_w / 2
        elements += f'<circle cx="{cx:.0f}" cy="{wr_y:.0f}" r="4" fill="#f59e0b"/>'
        if i > 0:
            prev_x = margin_l + (i-1) * gap + gap * 0.2 + bar_w / 2
            prev_wr_y = margin_t + (1 - results[i-1]["win_rate"]) * ph
            elements += f'<line x1="{prev_x:.0f}" y1="{prev_wr_y:.0f}" x2="{cx:.0f}" y2="{wr_y:.0f}" stroke="#f59e0b" stroke-width="2"/>'

    # Labels
    elements += f'<text x="{margin_l-5}" y="{margin_t+10}" text-anchor="end" font-size="10" fill="#16a34a">Sharpe</text>'
    elements += f'<text x="{w-margin_r+5}" y="{margin_t+10}" font-size="10" fill="#f59e0b">Win Rate</text>'
    elements += f'<text x="{w/2}" y="{h-3}" text-anchor="middle" font-size="11" fill="#334155">Threshold</text>'

    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{elements}</svg>'


def _svg_equity_curves(results: List[Dict], df: pd.DataFrame) -> str:
    """SVG equity curves for select thresholds."""
    w, h = 600, 250
    margin_l, margin_r, margin_t, margin_b = 50, 20, 25, 30
    pw = w - margin_l - margin_r
    ph = h - margin_t - margin_b

    oos = df.dropna(subset=["pred_prob"]).sort_values("entry_date")
    if len(oos) == 0:
        return ""

    show_thresholds = [0.50, 0.60, 0.70, 0.80]
    colors = ["#94a3b8", "#3b82f6", "#16a34a", "#f59e0b"]

    all_equities = {}
    for t in show_thresholds:
        sel = oos[oos["pred_prob"] >= t]
        if len(sel) == 0:
            continue
        eq = STARTING_CAPITAL + sel["pnl"].cumsum().values
        eq = np.concatenate([[STARTING_CAPITAL], eq])
        all_equities[t] = eq

    if not all_equities:
        return ""

    all_vals = np.concatenate(list(all_equities.values()))
    mn, mx = float(all_vals.min()), float(all_vals.max())
    rng = mx - mn if mx > mn else 1

    elements = ""
    elements += f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{h-margin_b}" stroke="#cbd5e1"/>'
    elements += f'<line x1="{margin_l}" y1="{h-margin_b}" x2="{w-margin_r}" y2="{h-margin_b}" stroke="#cbd5e1"/>'

    for ci, (t, eq) in enumerate(all_equities.items()):
        n = len(eq)
        points = []
        for i, v in enumerate(eq):
            x = margin_l + i / max(n - 1, 1) * pw
            y = margin_t + (1 - (v - mn) / rng) * ph
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        c = colors[ci % len(colors)]
        elements += f'<polyline points="{polyline}" fill="none" stroke="{c}" stroke-width="1.5"/>'
        # Legend
        lx = margin_l + 10 + ci * 120
        elements += f'<line x1="{lx}" y1="12" x2="{lx+20}" y2="12" stroke="{c}" stroke-width="2"/>'
        elements += f'<text x="{lx+25}" y="16" font-size="10" fill="#334155">≥{t:.2f}</text>'

    # Y labels
    elements += f'<text x="{margin_l-5}" y="{margin_t+5}" text-anchor="end" font-size="9" fill="#64748b">${mx:,.0f}</text>'
    elements += f'<text x="{margin_l-5}" y="{h-margin_b}" text-anchor="end" font-size="9" fill="#64748b">${mn:,.0f}</text>'

    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{elements}</svg>'


# ── Main ────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("EXP-710-max: Aggressive ML Signal Filtering Backtest")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"  {len(df)} trades, years {df['year'].min()}-{df['year'].max()}")
    print(f"  Base win rate: {df['win'].mean():.1%}")

    # Walk-forward training
    print("\n[2/4] Walk-forward ensemble training...")
    df = train_walk_forward(df)

    oos_count = df["pred_prob"].notna().sum()
    print(f"\n  OOS predictions: {oos_count} trades")

    # Backtest each threshold
    print("\n[3/4] Backtesting thresholds...")
    results = []
    for t in THRESHOLDS:
        r = backtest_threshold(df, t)
        results.append(r)
        print(f"  Threshold {t:.2f}: {r['n_trades']:3d} trades, "
              f"WR={r['win_rate']:.1%}, Sharpe={r['sharpe']:+.2f}, "
              f"P&L=${r['total_pnl']:+,.0f}, DD={r['max_drawdown']:.1%}")

    # Find optimal: highest Sharpe with >= 20 trades
    viable = [r for r in results if r["n_trades"] >= 20]
    if not viable:
        viable = [r for r in results if r["n_trades"] >= 5]
    if not viable:
        viable = results

    optimal = max(viable, key=lambda r: r["sharpe"])
    print(f"\n  >> Optimal threshold: {optimal['threshold']:.2f}")
    print(f"     Sharpe={optimal['sharpe']:.2f}, WR={optimal['win_rate']:.0%}, "
          f"Trades={optimal['n_trades']}, P&L=${optimal['total_pnl']:+,.0f}")

    # Save results
    print("\n[4/4] Generating outputs...")
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "experiment": "EXP-710-max",
        "description": "Aggressive ML Signal Filtering",
        "generated": datetime.now().isoformat(),
        "data": {
            "total_trades": len(df),
            "oos_trades": int(oos_count),
            "years": sorted([int(y) for y in df["year"].unique()]),
            "features": FEATURES,
        },
        "optimal": optimal,
        "all_thresholds": results,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote results/summary.json")

    html = generate_html(results, optimal, df)
    (RESULTS_DIR / "report.html").write_text(html)
    print(f"  Wrote results/report.html")

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
