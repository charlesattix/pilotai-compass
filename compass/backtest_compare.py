"""
compass/backtest_compare.py — Cross-experiment backtest comparison tool.

Loads trade-level data from backtest CSVs (EXP-400, EXP-401) and paper
DBs (EXP-503, EXP-600), computes side-by-side metrics, correlation
heatmap, and regime-conditioned performance, then generates a standalone
HTML report.

Usage::

    python3 -m compass.backtest_compare
    # → reports/backtest_comparison.html
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STARTING_CAPITAL = 100_000.0
REPORT_PATH = ROOT / "reports" / "backtest_comparison.html"

# ── Data sources ──────────────────────────────────────────────────────────

_CSV_SOURCES: Dict[str, Path] = {
    "EXP-400": ROOT / "compass" / "training_data_exp400.csv",
    "EXP-401": ROOT / "compass" / "training_data_exp401.csv",
}

_DB_SOURCES: Dict[str, Path] = {
    "EXP-503": ROOT / "data" / "exp503" / "pilotai_exp503.db",
    "EXP-600": ROOT / "data" / "exp600" / "pilotai_exp600.db",
}


# ── Trade-level loading ──────────────────────────────────────────────────


def load_trades(csv_path: Path) -> pd.DataFrame:
    """Load trade-level backtest data from a CSV."""
    df = pd.read_csv(csv_path, parse_dates=["entry_date", "exit_date"])
    if "year" not in df.columns and "exit_date" in df.columns:
        df["year"] = df["exit_date"].dt.year
    return df


def load_trades_from_db(db_path: Path) -> pd.DataFrame:
    """Load trade-level data from a paper-trading SQLite DB."""
    import sqlite3
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql("SELECT * FROM trades WHERE status = 'closed'", conn)
        conn.close()
        if df.empty:
            return df
        for col in ["entry_date", "exit_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        if "year" not in df.columns and "exit_date" in df.columns:
            df["year"] = df["exit_date"].dt.year
        return df
    except Exception as exc:
        logger.warning("Failed to load %s: %s", db_path, exc)
        return pd.DataFrame()


def load_all_trades() -> Dict[str, pd.DataFrame]:
    """Load trade data for all available experiments."""
    result: Dict[str, pd.DataFrame] = {}
    for name, path in _CSV_SOURCES.items():
        if path.exists():
            df = load_trades(path)
            if len(df) > 0:
                result[name] = df
                logger.info("%s: loaded %d trades from CSV", name, len(df))
    for name, path in _DB_SOURCES.items():
        if path.exists():
            df = load_trades_from_db(path)
            if len(df) > 0:
                result[name] = df
                logger.info("%s: loaded %d trades from DB", name, len(df))
    return result


# ── Metrics computation ──────────────────────────────────────────────────


@dataclass
class ExperimentMetrics:
    """Metrics for one experiment over one time period."""
    experiment: str
    period: str  # "2020", "2021", ..., "Overall"
    n_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    sharpe: float = 0.0
    annual_return_pct: float = 0.0
    # Regime breakdown (optional)
    regime: str = ""


def compute_metrics(
    trades: pd.DataFrame,
    period: str,
    experiment: str,
    starting_capital: float = STARTING_CAPITAL,
    regime: str = "",
) -> ExperimentMetrics:
    """Compute metrics for a set of trades."""
    n = len(trades)
    if n == 0:
        return ExperimentMetrics(experiment=experiment, period=period, regime=regime)

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    total_pnl = float(trades["pnl"].sum())
    gross_wins = float(wins["pnl"].sum()) if len(wins) > 0 else 0.0
    gross_losses = abs(float(losses["pnl"].sum())) if len(losses) > 0 else 0.0

    # Daily returns for Sharpe / drawdown
    daily_pnl = trades.groupby("exit_date")["pnl"].sum().sort_index()
    if len(daily_pnl) > 1:
        idx = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
        daily_pnl = daily_pnl.reindex(idx, fill_value=0.0)

    daily_returns = daily_pnl / starting_capital
    n_days = len(daily_returns)

    # Sharpe
    if n_days > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    equity = starting_capital + daily_pnl.cumsum()
    if len(equity) > 0:
        hwm = equity.cummax()
        dd = (equity - hwm) / hwm
        max_dd_pct = float(dd.min()) * 100
    else:
        max_dd_pct = 0.0

    # Annual return
    if n_days > 0:
        total_return = total_pnl / starting_capital
        years = max(n_days / 252, 0.01)
        if total_return > -1:
            annual_return_pct = ((1 + total_return) ** (1 / years) - 1) * 100
        else:
            annual_return_pct = -100.0
    else:
        annual_return_pct = 0.0

    return ExperimentMetrics(
        experiment=experiment,
        period=period,
        n_trades=n,
        win_rate=round(len(wins) / n * 100, 1) if n > 0 else 0.0,
        total_pnl=round(total_pnl, 2),
        avg_pnl=round(total_pnl / n, 2) if n > 0 else 0.0,
        avg_win=round(float(wins["pnl"].mean()), 2) if len(wins) > 0 else 0.0,
        avg_loss=round(float(losses["pnl"].mean()), 2) if len(losses) > 0 else 0.0,
        profit_factor=round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0.0,
        max_dd_pct=round(max_dd_pct, 1),
        sharpe=round(sharpe, 3),
        annual_return_pct=round(annual_return_pct, 1),
        regime=regime,
    )


def compute_all_metrics(
    all_trades: Dict[str, pd.DataFrame],
) -> Tuple[List[ExperimentMetrics], List[ExperimentMetrics], List[ExperimentMetrics]]:
    """Compute overall, per-year, and per-regime metrics for all experiments.

    Returns:
        (overall_metrics, yearly_metrics, regime_metrics)
    """
    overall: List[ExperimentMetrics] = []
    yearly: List[ExperimentMetrics] = []
    regime_list: List[ExperimentMetrics] = []

    for name, df in all_trades.items():
        # Overall
        overall.append(compute_metrics(df, "Overall", name))

        # Per year
        for year in sorted(df["year"].unique()):
            yr_df = df[df["year"] == year]
            yearly.append(compute_metrics(yr_df, str(int(year)), name))

        # Per regime
        if "regime" in df.columns:
            for regime in sorted(df["regime"].dropna().unique()):
                r_df = df[df["regime"] == regime]
                if len(r_df) >= 3:
                    regime_list.append(
                        compute_metrics(r_df, "Overall", name, regime=regime)
                    )

    return overall, yearly, regime_list


# ── Correlation ──────────────────────────────────────────────────────────


def build_daily_return_matrix(
    all_trades: Dict[str, pd.DataFrame],
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    """Build a DataFrame of aligned daily returns across experiments."""
    series_dict: Dict[str, pd.Series] = {}
    for name, df in all_trades.items():
        daily_pnl = df.groupby("exit_date")["pnl"].sum().sort_index()
        if len(daily_pnl) > 1:
            idx = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
            daily_pnl = daily_pnl.reindex(idx, fill_value=0.0)
        series_dict[name] = daily_pnl / starting_capital

    if not series_dict:
        return pd.DataFrame()

    matrix = pd.DataFrame(series_dict)
    matrix = matrix.fillna(0.0)
    return matrix


def compute_correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of daily returns."""
    if matrix.empty or matrix.shape[1] < 2:
        return pd.DataFrame()
    return matrix.corr()


# ── HTML report ──────────────────────────────────────────────────────────


def _esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_pnl(v: float) -> str:
    color = "#16a34a" if v > 0 else "#dc2626" if v < 0 else "#64748b"
    return f'<span style="color:{color}">${v:,.0f}</span>'


def _fmt_pct(v: float, color: bool = False) -> str:
    if color:
        c = "#16a34a" if v > 0 else "#dc2626" if v < 0 else "#64748b"
        return f'<span style="color:{c}">{v:+.1f}%</span>'
    return f"{v:.1f}%"


def _fmt_sharpe(v: float) -> str:
    if v >= 1.5:
        c = "#16a34a"
    elif v >= 0.8:
        c = "#ca8a04"
    elif v >= 0:
        c = "#64748b"
    else:
        c = "#dc2626"
    return f'<span style="color:{c};font-weight:600">{v:.3f}</span>'


def _metrics_table(
    metrics: List[ExperimentMetrics],
    experiments: List[str],
    group_col: str = "period",
    show_regime: bool = False,
) -> str:
    """Render a comparison table with experiments as columns."""
    # Collect unique group values
    if show_regime:
        groups = sorted(set(m.regime for m in metrics if m.regime))
    else:
        groups = sorted(set(getattr(m, group_col) for m in metrics))

    # Build lookup
    lookup: Dict[Tuple[str, str], ExperimentMetrics] = {}
    for m in metrics:
        key_val = m.regime if show_regime else getattr(m, group_col)
        lookup[(key_val, m.experiment)] = m

    header = f"<th>{'Regime' if show_regime else 'Period'}</th><th>Metric</th>"
    for exp in experiments:
        header += f"<th>{_esc(exp)}</th>"

    rows = ""
    metric_defs = [
        ("Trades", lambda m: str(m.n_trades)),
        ("Win Rate", lambda m: _fmt_pct(m.win_rate)),
        ("Total PnL", lambda m: _fmt_pnl(m.total_pnl)),
        ("Avg PnL", lambda m: _fmt_pnl(m.avg_pnl)),
        ("Profit Factor", lambda m: f"{m.profit_factor:.2f}"),
        ("Sharpe", lambda m: _fmt_sharpe(m.sharpe)),
        ("Max DD", lambda m: _fmt_pct(m.max_dd_pct, color=True)),
        ("Ann. Return", lambda m: _fmt_pct(m.annual_return_pct, color=True)),
    ]

    for g in groups:
        first = True
        for label, fmt_fn in metric_defs:
            row_label = f'<td rowspan="{len(metric_defs)}" style="font-weight:600;vertical-align:top;border-right:2px solid #e2e8f0">{_esc(g)}</td>' if first else ""
            cells = f"<td>{label}</td>"
            for exp in experiments:
                m = lookup.get((g, exp))
                if m and m.n_trades > 0:
                    cells += f"<td>{fmt_fn(m)}</td>"
                else:
                    cells += '<td style="color:#cbd5e1">-</td>'
            style = 'border-top:2px solid #e2e8f0' if first else ''
            rows += f'<tr style="{style}">{row_label}{cells}</tr>'
            first = False

    return (
        f'<table class="dt"><thead><tr>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _correlation_heatmap_html(corr: pd.DataFrame) -> str:
    """Render a correlation matrix as an inline HTML table with color coding."""
    if corr.empty:
        return "<p><em>Insufficient data for correlation analysis.</em></p>"

    labels = list(corr.columns)
    header = "<th></th>" + "".join(f"<th>{_esc(l)}</th>" for l in labels)
    rows = ""
    for r_label in labels:
        cells = f'<td style="font-weight:600">{_esc(r_label)}</td>'
        for c_label in labels:
            val = corr.loc[r_label, c_label]
            # Color: green (low corr = diversification) to red (high corr)
            if r_label == c_label:
                bg = "#f1f5f9"
                txt = "1.00"
            else:
                intensity = min(abs(val), 1.0)
                if val >= 0:
                    r, g, b = int(255 - intensity * 80), int(255 - intensity * 30), int(255 - intensity * 80)
                else:
                    r, g, b = int(220 + intensity * 35), int(255 - intensity * 60), int(220 + intensity * 35)
                bg = f"rgb({r},{g},{b})"
                txt = f"{val:.3f}"
            cells += f'<td style="background:{bg};text-align:center;font-weight:600;padding:10px">{txt}</td>'
        rows += f"<tr>{cells}</tr>"

    return (
        f'<table class="dt" style="width:auto">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def generate_html(
    all_trades: Dict[str, pd.DataFrame],
    overall: List[ExperimentMetrics],
    yearly: List[ExperimentMetrics],
    regime_metrics: List[ExperimentMetrics],
    corr_matrix: pd.DataFrame,
) -> str:
    """Generate the complete comparison HTML report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    experiments = sorted(all_trades.keys())
    n_total_trades = sum(len(df) for df in all_trades.values())

    # Summary cards
    cards = ""
    for exp in experiments:
        m = next((x for x in overall if x.experiment == exp), None)
        if m:
            cards += (
                f'<div class="card">'
                f'<div class="card-title">{_esc(exp)}</div>'
                f'<div class="card-value">{_fmt_sharpe(m.sharpe)}</div>'
                f'<div class="card-sub">{m.n_trades} trades | '
                f'WR {m.win_rate:.0f}% | {_fmt_pnl(m.total_pnl)}</div>'
                f'</div>'
            )

    # Tables
    overall_table = _metrics_table(overall, experiments, "period")
    yearly_table = _metrics_table(yearly, experiments, "period")
    regime_table = _metrics_table(regime_metrics, experiments, "regime", show_regime=True)
    corr_html = _correlation_heatmap_html(corr_matrix)

    # Best/worst per metric across experiments
    highlights = ""
    if overall:
        best_sharpe = max(overall, key=lambda m: m.sharpe)
        best_wr = max(overall, key=lambda m: m.win_rate)
        best_pf = max(overall, key=lambda m: m.profit_factor)
        least_dd = max(overall, key=lambda m: m.max_dd_pct)  # least negative
        highlights = (
            f'<div class="highlights">'
            f'Best Sharpe: <strong>{best_sharpe.experiment}</strong> ({best_sharpe.sharpe:.3f}) · '
            f'Best Win Rate: <strong>{best_wr.experiment}</strong> ({best_wr.win_rate:.0f}%) · '
            f'Best Profit Factor: <strong>{best_pf.experiment}</strong> ({best_pf.profit_factor:.2f}) · '
            f'Smallest DD: <strong>{least_dd.experiment}</strong> ({least_dd.max_dd_pct:.1f}%)'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Comparison Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.subtitle{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;
min-width:180px;flex:1;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card-title{{font-size:0.78em;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.card-value{{font-size:1.5em;font-weight:700}}
.card-sub{{font-size:0.8em;color:#94a3b8;margin-top:2px}}
.highlights{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:10px 16px;
font-size:0.88em;margin-bottom:20px}}
.dt{{border-collapse:collapse;width:100%;font-size:0.85em;margin-bottom:16px}}
.dt th{{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0}}
.dt td{{padding:6px 10px;border-bottom:1px solid #f1f5f9}}
.dt tr:hover{{background:#f8fafc}}
.note{{font-size:0.82em;color:#94a3b8;margin-bottom:8px}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style>
</head>
<body>

<h1>Backtest Comparison Report</h1>
<p class="subtitle">{len(experiments)} experiments · {n_total_trades:,} total trades · Generated {now}</p>

<div class="cards">{cards}</div>
{highlights}

<h2>Overall Performance</h2>
<p class="note">Metrics computed over each experiment's full backtest period.</p>
{overall_table}

<h2>Year-by-Year Comparison</h2>
<p class="note">Per-year metrics. Sharpe and max DD computed from daily returns within each year.</p>
{yearly_table}

<h2>Daily Return Correlations</h2>
<p class="note">Pearson correlation of daily returns across experiments.
Low correlation enables diversification; high correlation means redundant risk.</p>
{corr_html}

<h2>Regime-Conditioned Performance</h2>
<p class="note">Performance split by market regime (bull, bear, high_vol, crash, low_vol).
Minimum 3 trades per regime required.</p>
{regime_table}

<hr>
<p style="font-size:0.75em;color:#94a3b8">
Generated by <code>compass/backtest_compare.py</code> &mdash; {now}
</p>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    all_trades = load_all_trades()
    if not all_trades:
        logger.error("No trade data found")
        return

    overall, yearly, regime_metrics = compute_all_metrics(all_trades)
    ret_matrix = build_daily_return_matrix(all_trades)
    corr_matrix = compute_correlation(ret_matrix)

    html = generate_html(all_trades, overall, yearly, regime_metrics, corr_matrix)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    logger.info("Report → %s (%d bytes)", REPORT_PATH, len(html))


if __name__ == "__main__":
    main()
