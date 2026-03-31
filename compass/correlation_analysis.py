"""
Cross-experiment correlation analysis for Phase 5 portfolio optimization.

Loads daily P&L series from paper-trading SQLite databases and/or backtest
training-data CSVs, then computes:
  - Per-experiment risk metrics (Sharpe, Sortino, Calmar)
  - Pairwise return correlation matrix
  - Rolling 21-day correlation time series
  - Self-contained HTML report with heatmaps and charts

This is READ-ONLY analysis.  No broker connections, no trade placement.

Data sources (tried in order):
  1. Paper trading SQLite DBs (live data — sparse early on)
  2. Backtest training data CSVs (6 years of simulated trades)

Usage::

    from compass.correlation_analysis import generate_correlation_report
    generate_correlation_report()  # default: reports/correlation_analysis.html
"""

from __future__ import annotations

import base64
import io
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "correlation_analysis.html"
DEFAULT_CAPITAL = 100_000.0

# ── Daily P&L extraction ────────────────────────────────────────────────


def daily_pnl_from_db(db_path: Path, starting_capital: float = DEFAULT_CAPITAL) -> pd.Series:
    """Extract daily P&L series from a paper-trading SQLite database.

    Groups closed-trade PnL by exit_date, fills non-trading days with 0.

    Returns:
        pd.Series with DatetimeIndex, values = daily P&L in dollars.
        Empty Series if no usable data.
    """
    if not db_path.exists():
        return pd.Series(dtype=float)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT exit_date, pnl FROM trades "
            "WHERE pnl IS NOT NULL AND exit_date IS NOT NULL "
            "ORDER BY exit_date"
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to read %s: %s", db_path, exc)
        return pd.Series(dtype=float)

    if not rows:
        return pd.Series(dtype=float)

    by_date: Dict[str, float] = {}
    for r in rows:
        date_str = str(r["exit_date"])[:10]
        by_date[date_str] = by_date.get(date_str, 0.0) + r["pnl"]

    dates = sorted(by_date.keys())
    idx = pd.bdate_range(start=dates[0], end=dates[-1])
    series = pd.Series(0.0, index=idx, name="daily_pnl")
    for d, pnl in by_date.items():
        ts = pd.Timestamp(d)
        if ts in series.index:
            series.loc[ts] = pnl
    return series


def daily_pnl_from_csv(
    csv_path: Path,
    starting_capital: float = DEFAULT_CAPITAL,
) -> pd.Series:
    """Extract daily P&L series from a backtest training data CSV.

    Uses the 'exit_date' and 'pnl' columns.
    """
    if not csv_path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return pd.Series(dtype=float)

    if "exit_date" not in df.columns or "pnl" not in df.columns:
        return pd.Series(dtype=float)

    daily = df.groupby("exit_date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Fill to business day calendar
    if len(daily) > 1:
        full_idx = pd.bdate_range(start=daily.index.min(), end=daily.index.max())
        daily = daily.reindex(full_idx, fill_value=0.0)
    daily.name = "daily_pnl"
    return daily


# ── Data source configuration ────────────────────────────────────────────

# Maps experiment labels to their data sources.
# Priority: paper DB first, backtest CSV fallback.
_DEFAULT_SOURCES = {
    "EXP-400": {
        "db": ROOT / "data" / "pilotai.db",
        "csv": ROOT / "compass" / "training_data_exp400.csv",
        "ticker": "SPY",
    },
    "EXP-401": {
        "db": ROOT / "data" / "pilotai_exp401.db",
        "csv": ROOT / "compass" / "training_data_exp401.csv",
        "ticker": "SPY",
    },
    "EXP-503": {
        "db": ROOT / "data" / "exp503" / "pilotai_exp503.db",
        "csv": None,
        "ticker": "SPY",
    },
    "EXP-600": {
        "db": ROOT / "data" / "exp600" / "pilotai_exp600.db",
        "csv": None,
        "ticker": "IBIT",
    },
}


def load_all_daily_pnl(
    sources: Optional[Dict[str, Dict]] = None,
) -> Dict[str, pd.Series]:
    """Load daily P&L for all experiments.

    Tries paper DB first; if that yields < 5 data points, falls back to CSV.

    Returns:
        {experiment_label: daily_pnl_series}  (only experiments with data)
    """
    if sources is None:
        sources = _DEFAULT_SOURCES

    result: Dict[str, pd.Series] = {}
    for label, src in sources.items():
        series = pd.Series(dtype=float)

        # Try paper DB
        db_path = src.get("db")
        if db_path and Path(db_path).exists():
            series = daily_pnl_from_db(Path(db_path))
            if len(series) >= 5:
                logger.info("%s: loaded %d days from paper DB", label, len(series))

        # Fallback to CSV
        if len(series) < 5:
            csv_path = src.get("csv")
            if csv_path and Path(csv_path).exists():
                series = daily_pnl_from_csv(Path(csv_path))
                if len(series) > 0:
                    logger.info("%s: loaded %d days from backtest CSV", label, len(series))

        if len(series) > 0:
            series.name = label
            result[label] = series
        else:
            logger.info("%s: no data available", label)

    return result


# ── Risk metrics ─────────────────────────────────────────────────────────


def compute_risk_metrics(
    daily_pnl: pd.Series,
    starting_capital: float = DEFAULT_CAPITAL,
    annual_factor: float = 252.0,
) -> Dict[str, Optional[float]]:
    """Compute Sharpe, Sortino, Calmar, and other risk metrics.

    Args:
        daily_pnl: Daily P&L in dollars.
        starting_capital: For return conversion and drawdown calculation.
        annual_factor: Trading days per year.

    Returns:
        Dict with sharpe, sortino, calmar, annual_return_pct,
        annual_vol_pct, max_drawdown_pct, total_return_pct, n_days,
        win_rate_daily (fraction of days with positive P&L).
    """
    n = len(daily_pnl)
    if n < 2:
        return {
            "sharpe": None, "sortino": None, "calmar": None,
            "annual_return_pct": None, "annual_vol_pct": None,
            "max_drawdown_pct": None, "total_return_pct": None,
            "n_days": n, "win_rate_daily": None,
        }

    returns = daily_pnl / starting_capital  # daily return fraction
    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))

    # Sharpe
    sharpe = (mean_r / std_r * math.sqrt(annual_factor)) if std_r > 0 else None

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / downside_std * math.sqrt(annual_factor)) if downside_std > 0 else None

    # Equity curve and drawdown
    equity = starting_capital + daily_pnl.cumsum()
    hwm = equity.cummax()
    drawdown = (equity - hwm) / hwm
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # CAGR for Calmar
    total_return = float(equity.iloc[-1] / starting_capital - 1)
    years = n / annual_factor
    if total_return > -1 and years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else None

    win_rate = float((daily_pnl > 0).sum() / n)

    return {
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "sortino": round(sortino, 3) if sortino is not None else None,
        "calmar": round(calmar, 3) if calmar is not None else None,
        "annual_return_pct": round(cagr * 100, 2),
        "annual_vol_pct": round(std_r * math.sqrt(annual_factor) * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "n_days": n,
        "win_rate_daily": round(win_rate, 4),
    }


# ── Correlation computation ──────────────────────────────────────────────


def build_return_matrix(
    daily_pnls: Dict[str, pd.Series],
    starting_capital: float = DEFAULT_CAPITAL,
) -> pd.DataFrame:
    """Align daily P&L series into a return matrix on common dates.

    Returns DataFrame with DatetimeIndex, columns = experiment labels,
    values = daily return fraction.  Only dates present in ALL experiments.
    """
    if not daily_pnls:
        return pd.DataFrame()

    frames = {}
    for label, series in daily_pnls.items():
        frames[label] = series / starting_capital

    df = pd.DataFrame(frames)
    df = df.dropna()  # keep only dates with data for all experiments
    return df


def compute_correlation_matrix(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix across experiments."""
    if return_matrix.empty or len(return_matrix) < 5:
        return pd.DataFrame()
    return return_matrix.corr()


def compute_rolling_correlation(
    return_matrix: pd.DataFrame,
    window: int = 21,
) -> Dict[str, pd.Series]:
    """Rolling pairwise correlations for all unique experiment pairs.

    Returns:
        {pair_label: pd.Series of rolling correlation}
    """
    cols = list(return_matrix.columns)
    result: Dict[str, pd.Series] = {}

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = f"{cols[i]} vs {cols[j]}"
            rolling_corr = return_matrix[cols[i]].rolling(window).corr(return_matrix[cols[j]])
            rolling_corr = rolling_corr.dropna()
            if len(rolling_corr) > 0:
                result[pair] = rolling_corr

    return result


# ── Chart rendering ──────────────────────────────────────────────────────


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return b64


def _render_correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    """Render correlation matrix as an annotated heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * n), max(4, 1.5 * n)))

    im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_matrix.index, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Return Correlation Matrix (full period)", fontsize=12, pad=12)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_rolling_correlations(
    rolling_corrs: Dict[str, pd.Series],
    window: int,
) -> str:
    """Line chart of rolling pairwise correlations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rolling_corrs:
        return ""

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rolling_corrs), 1)))

    for idx, (pair, series) in enumerate(rolling_corrs.items()):
        ax.plot(series.index, series.values, label=pair, lw=1.5, color=colors[idx])

    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.3)
    ax.axhline(-0.5, color="gray", lw=0.5, ls="--", alpha=0.3)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Rolling {window}-Day Pairwise Correlation", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_equity_curves(
    daily_pnls: Dict[str, pd.Series],
    starting_capital: float,
) -> str:
    """Cumulative equity curves per experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(daily_pnls), 1)))

    for idx, (label, series) in enumerate(daily_pnls.items()):
        equity = starting_capital + series.cumsum()
        ret_pct = (equity / starting_capital - 1) * 100
        ax.plot(ret_pct.index, ret_pct.values, label=label, lw=1.5, color=colors[idx])

    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Equity Curves (cumulative % return)", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── HTML report ──────────────────────────────────────────────────────────


def _fmt(val, fmt_str=".2f", prefix="", suffix="") -> str:
    if val is None:
        return "—"
    try:
        return f"{prefix}{val:{fmt_str}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def generate_html(
    daily_pnls: Dict[str, pd.Series],
    metrics: Dict[str, Dict],
    corr_matrix: pd.DataFrame,
    rolling_corrs: Dict[str, pd.Series],
    heatmap_b64: str,
    rolling_b64: str,
    equity_b64: str,
    window: int,
) -> str:
    """Assemble the self-contained HTML report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_experiments = len(daily_pnls)

    # Data source badges
    data_notes = []
    for label, series in daily_pnls.items():
        n = len(series)
        start = str(series.index.min().date()) if n > 0 else "?"
        end = str(series.index.max().date()) if n > 0 else "?"
        data_notes.append(f"<strong>{label}</strong>: {n} days ({start} to {end})")

    # Risk metrics table
    metrics_rows = ""
    labels = sorted(metrics.keys())
    for label in labels:
        m = metrics[label]
        sharpe_cls = "good" if m.get("sharpe") and m["sharpe"] > 1.0 else ""
        dd_cls = "bad" if m.get("max_drawdown_pct") and m["max_drawdown_pct"] < -20 else ""
        metrics_rows += (
            f'<tr>'
            f'<td><strong>{label}</strong></td>'
            f'<td class="{sharpe_cls}">{_fmt(m["sharpe"], ".3f")}</td>'
            f'<td>{_fmt(m["sortino"], ".3f")}</td>'
            f'<td>{_fmt(m["calmar"], ".3f")}</td>'
            f'<td>{_fmt(m["annual_return_pct"], ".2f", suffix="%")}</td>'
            f'<td>{_fmt(m["annual_vol_pct"], ".2f", suffix="%")}</td>'
            f'<td class="{dd_cls}">{_fmt(m["max_drawdown_pct"], ".2f", suffix="%")}</td>'
            f'<td>{_fmt(m["total_return_pct"], ".2f", suffix="%")}</td>'
            f'<td>{m["n_days"]}</td>'
            f'<td>{_fmt(m["win_rate_daily"], ".1%")}</td>'
            f'</tr>\n'
        )

    # Correlation summary for portfolio optimization takeaways
    takeaways = []
    if not corr_matrix.empty:
        pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
        pairs.sort(key=lambda x: x[2])

        if pairs:
            low = pairs[0]
            high = pairs[-1]
            takeaways.append(f"Lowest correlation: <strong>{low[0]}</strong> vs <strong>{low[1]}</strong> ({low[2]:.2f}) — best diversification pair")
            takeaways.append(f"Highest correlation: <strong>{high[0]}</strong> vs <strong>{high[1]}</strong> ({high[2]:.2f}) — most redundant pair")

            avg_corr = np.mean([p[2] for p in pairs])
            takeaways.append(f"Average pairwise correlation: <strong>{avg_corr:.2f}</strong>")

            if avg_corr < 0.3:
                takeaways.append("Portfolio diversification potential: <span class='good'>HIGH</span> — experiments have low overlap")
            elif avg_corr < 0.6:
                takeaways.append("Portfolio diversification potential: <span class='ok'>MODERATE</span>")
            else:
                takeaways.append("Portfolio diversification potential: <span class='bad'>LOW</span> — experiments are highly correlated")

    takeaways_html = "".join(f"<li>{t}</li>" for t in takeaways) if takeaways else "<li>Insufficient overlapping data for correlation analysis</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Correlation Analysis — Portfolio Optimization Prep</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2.5em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .data-sources {{ background: #f1f5f9; border-radius: 6px; padding: 1em;
                   font-size: 0.85em; margin: 1em 0; }}
  .data-sources p {{ margin: 0.3em 0; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .ok {{ color: #d97706; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 12px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  tr:hover {{ background: #f8fafc; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .takeaways {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 1.2em 1.5em; margin: 1.5em 0; }}
  .takeaways ul {{ margin: 0.5em 0; padding-left: 1.5em; }}
  .takeaways li {{ margin: 0.4em 0; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Correlation Analysis</h1>
<div class="meta">
  Phase 5 portfolio optimization prep &middot;
  <strong>{n_experiments}</strong> experiments &middot;
  Generated {now}
</div>

<div class="data-sources">
  <strong>Data Sources:</strong>
  {"<br>".join(data_notes)}
</div>

<h2>1. Risk Metrics Summary</h2>
<table>
<thead>
<tr>
  <th>Experiment</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th>
  <th>Ann. Return</th><th>Ann. Vol</th><th>Max DD</th>
  <th>Total Return</th><th>Days</th><th>Win Rate (daily)</th>
</tr>
</thead>
<tbody>
{metrics_rows}
</tbody>
</table>

<h2>2. Equity Curves</h2>
<div class="chart">
  <img src="data:image/png;base64,{equity_b64}" alt="Equity Curves">
</div>

<h2>3. Correlation Matrix</h2>
{"<div class='chart'><img src='data:image/png;base64," + heatmap_b64 + "' alt='Correlation Heatmap'></div>" if heatmap_b64 else "<p class='meta'>Insufficient overlapping data for correlation matrix (need ≥ 2 experiments with ≥ 5 common dates).</p>"}

<h2>4. Rolling {window}-Day Correlation</h2>
{"<div class='chart'><img src='data:image/png;base64," + rolling_b64 + "' alt='Rolling Correlation'></div>" if rolling_b64 else "<p class='meta'>Insufficient overlapping data for rolling correlation.</p>"}

<h2>5. Portfolio Optimization Takeaways</h2>
<div class="takeaways">
  <ul>{takeaways_html}</ul>
</div>

<footer>
  Generated by <code>compass/correlation_analysis.py</code> &middot;
  READ-ONLY analysis of existing trade data
</footer>

</body>
</html>"""

    return html


# ── Public API ───────────────────────────────────────────────────────────


def generate_correlation_report(
    output: str = str(DEFAULT_OUTPUT),
    sources: Optional[Dict[str, Dict]] = None,
    starting_capital: float = DEFAULT_CAPITAL,
    rolling_window: int = 21,
) -> str:
    """Generate the full correlation analysis HTML report.

    Args:
        output: Path for the HTML file.
        sources: Override data sources (for testing).
        starting_capital: For return conversion.
        rolling_window: Days for rolling correlation.

    Returns:
        Absolute path to the generated report.
    """
    logger.info("Loading daily P&L for all experiments...")
    daily_pnls = load_all_daily_pnl(sources)
    logger.info("Loaded data for %d experiments", len(daily_pnls))

    if not daily_pnls:
        logger.warning("No experiment data found — generating empty report")

    # Risk metrics
    metrics = {}
    for label, series in daily_pnls.items():
        metrics[label] = compute_risk_metrics(series, starting_capital)
        logger.info("%s: Sharpe=%s, Sortino=%s, MaxDD=%s",
                    label, metrics[label]["sharpe"],
                    metrics[label]["sortino"], metrics[label]["max_drawdown_pct"])

    # Correlation
    return_matrix = build_return_matrix(daily_pnls, starting_capital)
    corr_matrix = compute_correlation_matrix(return_matrix)
    rolling_corrs = compute_rolling_correlation(return_matrix, rolling_window)

    # Charts
    logger.info("Rendering charts...")
    heatmap_b64 = _render_correlation_heatmap(corr_matrix) if not corr_matrix.empty else ""
    rolling_b64 = _render_rolling_correlations(rolling_corrs, rolling_window) if rolling_corrs else ""
    equity_b64 = _render_equity_curves(daily_pnls, starting_capital) if daily_pnls else ""

    # HTML
    html = generate_html(
        daily_pnls, metrics, corr_matrix, rolling_corrs,
        heatmap_b64, rolling_b64, equity_b64, rolling_window,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Report written to %s (%d bytes)", out, len(html))

    return str(out.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = generate_correlation_report()
    print(f"Report: {path}")
