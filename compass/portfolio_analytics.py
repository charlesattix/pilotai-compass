"""Comprehensive portfolio analytics module.

Provides portfolio performance metrics, rolling analytics, benchmark
comparisons, risk contribution analysis, return tables, drawdown
detection, and self-contained HTML report generation.
"""

from __future__ import annotations

import datetime
import html as html_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class DrawdownPeriod:
    """Represents a single drawdown episode."""
    start: pd.Timestamp
    trough: pd.Timestamp
    end: Optional[pd.Timestamp]  # None if still in drawdown
    depth: float  # negative fraction
    recovery_days: Optional[int]  # None if not recovered


@dataclass
class PortfolioMetrics:
    """Summary statistics for a return series."""
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    avg_drawdown: float
    avg_recovery_days: float
    volatility: float
    skewness: float
    kurtosis: float


@dataclass
class RollingResult:
    """Container for rolling analytics series."""
    rolling_sharpe_30: pd.Series
    rolling_sharpe_60: pd.Series
    rolling_sharpe_90: pd.Series
    rolling_corr_30: Optional[pd.Series] = None
    rolling_corr_60: Optional[pd.Series] = None
    rolling_corr_90: Optional[pd.Series] = None


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of portfolio vs a benchmark."""
    name: str
    portfolio_metrics: PortfolioMetrics
    benchmark_metrics: PortfolioMetrics
    excess_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    correlation: float


@dataclass
class RiskContribution:
    """Per-strategy risk contribution."""
    strategy: str
    weight: float
    marginal_contribution: float
    percent_contribution: float
    standalone_vol: float


@dataclass
class ReturnTable:
    """Monthly / quarterly / annual return table."""
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    annual: pd.Series


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _annualisation_factor(daily_returns: pd.Series) -> float:
    """Estimate trading days per year from the index."""
    if len(daily_returns) < 2:
        return 252.0
    idx = daily_returns.index
    total_days = (idx[-1] - idx[0]).days
    if total_days == 0:
        return 252.0
    return len(daily_returns) / (total_days / 365.25)


def _to_series(returns: pd.Series) -> pd.Series:
    """Ensure we have a clean float Series with a DatetimeIndex."""
    s = returns.copy().astype(float).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()


def compute_total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def compute_cagr(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    idx = returns.index
    years = (idx[-1] - idx[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1)


def compute_volatility(returns: pd.Series, ann: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(ann))


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0,
                   ann: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / ann
    vol = excess.std(ddof=1)
    if vol < 1e-14:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(ann))


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0,
                    ann: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / ann
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    down_std = float(np.sqrt((downside ** 2).mean()))
    if down_std == 0:
        return 0.0
    return float(excess.mean() / down_std * np.sqrt(ann))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Return max drawdown as a negative fraction."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    return float(dd.min())


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1


def detect_drawdowns(returns: pd.Series) -> List[DrawdownPeriod]:
    """Identify all drawdown periods with recovery information."""
    dd = compute_drawdown_series(returns)
    periods: List[DrawdownPeriod] = []
    in_dd = False
    start = dd.index[0]
    trough = dd.index[0]
    trough_val = 0.0

    for i, (dt, val) in enumerate(dd.items()):
        if not in_dd:
            if val < 0:
                in_dd = True
                start = dt
                trough = dt
                trough_val = val
        else:
            if val < trough_val:
                trough = dt
                trough_val = val
            if val >= 0:
                in_dd = False
                rec_days = (dt - trough).days
                periods.append(DrawdownPeriod(
                    start=start, trough=trough, end=dt,
                    depth=trough_val, recovery_days=rec_days,
                ))

    if in_dd:
        periods.append(DrawdownPeriod(
            start=start, trough=trough, end=None,
            depth=trough_val, recovery_days=None,
        ))

    return periods


def compute_avg_drawdown(returns: pd.Series) -> float:
    periods = detect_drawdowns(returns)
    if not periods:
        return 0.0
    return float(np.mean([p.depth for p in periods]))


def compute_avg_recovery_days(returns: pd.Series) -> float:
    periods = detect_drawdowns(returns)
    recovered = [p.recovery_days for p in periods if p.recovery_days is not None]
    if not recovered:
        return float("nan")
    return float(np.mean(recovered))


def compute_calmar(returns: pd.Series) -> float:
    cagr = compute_cagr(returns)
    mdd = compute_max_drawdown(returns)
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / abs(mdd))


def compute_omega(returns: pd.Series, threshold: float = 0.0) -> float:
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def compute_skewness(returns: pd.Series) -> float:
    n = len(returns)
    if n < 3:
        return 0.0
    m = returns.mean()
    s = returns.std(ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * ((returns - m) ** 3).sum() / s ** 3)


def compute_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis."""
    n = len(returns)
    if n < 4:
        return 0.0
    m = returns.mean()
    s = returns.std(ddof=1)
    if s == 0:
        return 0.0
    m4 = ((returns - m) ** 4).mean()
    return float(m4 / s ** 4 - 3)


def build_portfolio_metrics(returns: pd.Series,
                            risk_free_rate: float = 0.0) -> PortfolioMetrics:
    ann = _annualisation_factor(returns)
    return PortfolioMetrics(
        total_return=compute_total_return(returns),
        cagr=compute_cagr(returns),
        sharpe=compute_sharpe(returns, risk_free_rate, ann),
        sortino=compute_sortino(returns, risk_free_rate, ann),
        calmar=compute_calmar(returns),
        omega=compute_omega(returns),
        max_drawdown=compute_max_drawdown(returns),
        avg_drawdown=compute_avg_drawdown(returns),
        avg_recovery_days=compute_avg_recovery_days(returns),
        volatility=compute_volatility(returns, ann),
        skewness=compute_skewness(returns),
        kurtosis=compute_kurtosis(returns),
    )


# ---------------------------------------------------------------------------
# Rolling analytics
# ---------------------------------------------------------------------------

def rolling_sharpe(returns: pd.Series, window: int,
                   risk_free_rate: float = 0.0) -> pd.Series:
    ann = _annualisation_factor(returns)
    rf_daily = risk_free_rate / ann
    excess = returns - rf_daily
    roll_mean = excess.rolling(window, min_periods=window).mean()
    roll_std = excess.rolling(window, min_periods=window).std(ddof=1)
    result = roll_mean / roll_std * np.sqrt(ann)
    return result.dropna()


def rolling_correlation(returns: pd.Series, benchmark: pd.Series,
                        window: int) -> pd.Series:
    aligned = pd.DataFrame({"p": returns, "b": benchmark}).dropna()
    return aligned["p"].rolling(window, min_periods=window).corr(
        aligned["b"]
    ).dropna()


def compute_rolling(returns: pd.Series,
                    benchmark: Optional[pd.Series] = None,
                    risk_free_rate: float = 0.0) -> RollingResult:
    rs30 = rolling_sharpe(returns, 30, risk_free_rate)
    rs60 = rolling_sharpe(returns, 60, risk_free_rate)
    rs90 = rolling_sharpe(returns, 90, risk_free_rate)
    rc30 = rc60 = rc90 = None
    if benchmark is not None:
        rc30 = rolling_correlation(returns, benchmark, 30)
        rc60 = rolling_correlation(returns, benchmark, 60)
        rc90 = rolling_correlation(returns, benchmark, 90)
    return RollingResult(
        rolling_sharpe_30=rs30, rolling_sharpe_60=rs60,
        rolling_sharpe_90=rs90,
        rolling_corr_30=rc30, rolling_corr_60=rc60, rolling_corr_90=rc90,
    )


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def compare_benchmark(returns: pd.Series, benchmark: pd.Series,
                      name: str = "Benchmark",
                      risk_free_rate: float = 0.0) -> BenchmarkComparison:
    aligned = pd.DataFrame({"p": returns, "b": benchmark}).dropna()
    p, b = aligned["p"], aligned["b"]
    p_metrics = build_portfolio_metrics(p, risk_free_rate)
    b_metrics = build_portfolio_metrics(b, risk_free_rate)
    excess = p - b
    te = float(excess.std(ddof=1) * np.sqrt(_annualisation_factor(p)))
    ir = float(excess.mean() / excess.std(ddof=1) * np.sqrt(
        _annualisation_factor(p))) if excess.std(ddof=1) > 0 else 0.0
    cov_mat = np.cov(p.values, b.values, ddof=1)
    beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] != 0 else 0.0
    ann = _annualisation_factor(p)
    alpha = float((p.mean() - risk_free_rate / ann
                   - beta * (b.mean() - risk_free_rate / ann)) * ann)
    corr = float(np.corrcoef(p.values, b.values)[0, 1])
    return BenchmarkComparison(
        name=name,
        portfolio_metrics=p_metrics,
        benchmark_metrics=b_metrics,
        excess_return=p_metrics.total_return - b_metrics.total_return,
        tracking_error=te,
        information_ratio=ir,
        beta=beta,
        alpha=alpha,
        correlation=corr,
    )


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------

def compute_risk_contributions(
    strategy_returns: Dict[str, pd.Series],
) -> List[RiskContribution]:
    """Compute marginal and percent risk contributions per strategy.

    Assumes equal-weight combination of strategies.
    """
    names = sorted(strategy_returns.keys())
    df = pd.DataFrame({n: strategy_returns[n] for n in names}).dropna()
    n = len(names)
    if n == 0 or len(df) < 2:
        return []
    weights = np.ones(n) / n
    cov = df.cov().values
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-12
    marginals = cov @ weights / port_vol
    contribs = weights * marginals
    total_contrib = contribs.sum()

    results = []
    for i, name in enumerate(names):
        results.append(RiskContribution(
            strategy=name,
            weight=float(weights[i]),
            marginal_contribution=float(marginals[i]),
            percent_contribution=float(
                contribs[i] / total_contrib) if total_contrib != 0 else 0.0,
            standalone_vol=float(df[name].std(ddof=1) * np.sqrt(252)),
        ))
    return results


# ---------------------------------------------------------------------------
# Return tables
# ---------------------------------------------------------------------------

def build_return_tables(returns: pd.Series) -> ReturnTable:
    """Build monthly, quarterly, and annual return tables."""
    s = _to_series(returns)

    # Monthly returns pivot (Year x Month)
    monthly = s.groupby([s.index.year, s.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["Year", "Month"])
    monthly_pivot = monthly.unstack(level="Month")
    monthly_pivot.columns = [
        datetime.date(2000, m, 1).strftime("%b") for m in monthly_pivot.columns
    ]

    # Quarterly
    quarterly = s.groupby([s.index.year, s.index.quarter]).apply(
        lambda x: (1 + x).prod() - 1
    )
    quarterly.index = pd.MultiIndex.from_tuples(
        quarterly.index, names=["Year", "Quarter"]
    )
    quarterly_pivot = quarterly.unstack(level="Quarter")
    quarterly_pivot.columns = [f"Q{q}" for q in quarterly_pivot.columns]

    # Annual
    annual = s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
    annual.index.name = "Year"

    return ReturnTable(
        monthly=monthly_pivot,
        quarterly=quarterly_pivot,
        annual=annual,
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',Tahoma,sans-serif;
margin:0;padding:20px 40px;}
h1{color:#00d4ff;border-bottom:2px solid #00d4ff;padding-bottom:10px;}
h2{color:#7ec8e3;margin-top:40px;}
table{border-collapse:collapse;width:100%;margin:15px 0;}
th,td{border:1px solid #333;padding:8px 12px;text-align:right;}
th{background:#16213e;color:#00d4ff;}
td{background:#0f3460;}
tr:nth-child(even) td{background:#1a1a40;}
.positive{color:#00e676;} .negative{color:#ff5252;}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
gap:15px;margin:15px 0;}
.metric-card{background:#16213e;border:1px solid #333;border-radius:8px;
padding:15px;text-align:center;}
.metric-card .label{font-size:0.85em;color:#7ec8e3;}
.metric-card .value{font-size:1.4em;font-weight:bold;margin-top:5px;}
svg{max-width:100%;}
.section{margin-bottom:40px;}
"""


def _fmt_pct(v: float) -> str:
    if np.isnan(v) or np.isinf(v):
        return f"{v}"
    return f"{v:+.2%}"


def _fmt_num(v: float, decimals: int = 3) -> str:
    if np.isnan(v) or np.isinf(v):
        return f"{v}"
    return f"{v:.{decimals}f}"


def _color_class(v: float) -> str:
    if np.isnan(v) or np.isinf(v):
        return ""
    return "positive" if v >= 0 else "negative"


def _svg_line_chart(series_dict: Dict[str, pd.Series],
                    width: int = 900, height: int = 300,
                    title: str = "") -> str:
    """Create a simple inline SVG line chart."""
    margin = {"top": 30, "right": 20, "bottom": 40, "left": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_vals: List[float] = []
    all_dates: List[pd.Timestamp] = []
    for s in series_dict.values():
        all_vals.extend(s.values.tolist())
        all_dates.extend(s.index.tolist())

    if not all_vals:
        return ""

    y_min, y_max = min(all_vals), max(all_vals)
    y_range = y_max - y_min if y_max != y_min else 1.0
    d_min = min(all_dates)
    d_max = max(all_dates)
    d_range = (d_max - d_min).total_seconds() or 1.0

    colors = ["#00d4ff", "#ff5252", "#00e676", "#ffab40", "#e040fb"]

    lines_svg = []
    for idx, (label, s) in enumerate(series_dict.items()):
        color = colors[idx % len(colors)]
        points = []
        for dt, val in s.items():
            x = margin["left"] + ((dt - d_min).total_seconds() / d_range) * plot_w
            y = margin["top"] + plot_h - ((val - y_min) / y_range) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5" />'
        legend_x = margin["left"] + 10 + idx * 150
        legend = (
            f'<rect x="{legend_x}" y="{height - 15}" width="12" height="12" fill="{color}" />'
            f'<text x="{legend_x + 16}" y="{height - 5}" fill="#e0e0e0" font-size="11">'
            f'{html_mod.escape(label)}</text>'
        )
        lines_svg.append(polyline + legend)

    # Y-axis labels
    y_labels = ""
    for i in range(5):
        val = y_min + y_range * i / 4
        y = margin["top"] + plot_h - (plot_h * i / 4)
        y_labels += (
            f'<text x="{margin["left"] - 5}" y="{y + 4}" fill="#888" '
            f'font-size="10" text-anchor="end">{val:.3f}</text>'
            f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" '
            f'y2="{y}" stroke="#333" stroke-width="0.5" />'
        )

    title_svg = ""
    if title:
        title_svg = (
            f'<text x="{width / 2}" y="18" fill="#00d4ff" font-size="14" '
            f'text-anchor="middle" font-weight="bold">{html_mod.escape(title)}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#1a1a2e" />'
        f'{title_svg}{y_labels}'
        f'{"".join(lines_svg)}'
        f'</svg>'
    )


def _metrics_cards_html(metrics: PortfolioMetrics) -> str:
    cards = [
        ("Total Return", _fmt_pct(metrics.total_return), metrics.total_return),
        ("CAGR", _fmt_pct(metrics.cagr), metrics.cagr),
        ("Sharpe", _fmt_num(metrics.sharpe), metrics.sharpe),
        ("Sortino", _fmt_num(metrics.sortino), metrics.sortino),
        ("Calmar", _fmt_num(metrics.calmar), metrics.calmar),
        ("Omega", _fmt_num(metrics.omega), metrics.omega),
        ("Max Drawdown", _fmt_pct(metrics.max_drawdown), metrics.max_drawdown),
        ("Avg Drawdown", _fmt_pct(metrics.avg_drawdown), metrics.avg_drawdown),
        ("Avg Recovery (days)", _fmt_num(metrics.avg_recovery_days, 0),
         metrics.avg_recovery_days),
        ("Volatility", _fmt_pct(metrics.volatility), metrics.volatility),
        ("Skewness", _fmt_num(metrics.skewness), metrics.skewness),
        ("Kurtosis", _fmt_num(metrics.kurtosis), metrics.kurtosis),
    ]
    inner = ""
    for label, value, raw in cards:
        cls = _color_class(raw) if isinstance(raw, float) else ""
        inner += (
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value {cls}">{value}</div></div>'
        )
    return f'<div class="metric-grid">{inner}</div>'


def _df_to_html(df: pd.DataFrame, pct: bool = True) -> str:
    """Convert a DataFrame to an HTML table with colouring."""
    rows = []
    header = "<tr><th></th>" + "".join(
        f"<th>{c}</th>" for c in df.columns
    ) + "</tr>"
    for idx, row in df.iterrows():
        cells = f"<td><b>{idx}</b></td>"
        for val in row:
            if pd.isna(val):
                cells += "<td>-</td>"
            else:
                cls = _color_class(val)
                s = _fmt_pct(val) if pct else _fmt_num(val)
                cells += f'<td class="{cls}">{s}</td>'
        rows.append(f"<tr>{cells}</tr>")
    return f"<table>{header}{''.join(rows)}</table>"


def _benchmark_table_html(comparisons: List[BenchmarkComparison]) -> str:
    header = (
        "<tr><th>Metric</th><th>Portfolio</th>"
        + "".join(f"<th>{html_mod.escape(c.name)}</th>" for c in comparisons)
        + "</tr>"
    )
    if not comparisons:
        return ""
    pm = comparisons[0].portfolio_metrics
    metric_rows = [
        ("Total Return", pm.total_return,
         [c.benchmark_metrics.total_return for c in comparisons]),
        ("CAGR", pm.cagr, [c.benchmark_metrics.cagr for c in comparisons]),
        ("Sharpe", pm.sharpe, [c.benchmark_metrics.sharpe for c in comparisons]),
        ("Sortino", pm.sortino, [c.benchmark_metrics.sortino for c in comparisons]),
        ("Max DD", pm.max_drawdown,
         [c.benchmark_metrics.max_drawdown for c in comparisons]),
        ("Volatility", pm.volatility,
         [c.benchmark_metrics.volatility for c in comparisons]),
    ]
    rows_html = ""
    for label, port_val, bench_vals in metric_rows:
        fmt = _fmt_pct if label in ("Total Return", "CAGR", "Max DD", "Volatility") else _fmt_num
        cells = f'<td><b>{label}</b></td><td class="{_color_class(port_val)}">{fmt(port_val)}</td>'
        for bv in bench_vals:
            cells += f'<td class="{_color_class(bv)}">{fmt(bv)}</td>'
        rows_html += f"<tr>{cells}</tr>"

    # Extra rows for alpha/beta/IR
    extra = [
        ("Alpha", [c.alpha for c in comparisons]),
        ("Beta", [c.beta for c in comparisons]),
        ("Info Ratio", [c.information_ratio for c in comparisons]),
        ("Correlation", [c.correlation for c in comparisons]),
    ]
    for label, vals in extra:
        cells = f"<td><b>{label}</b></td><td>-</td>"
        for v in vals:
            cells += f'<td class="{_color_class(v)}">{_fmt_num(v)}</td>'
        rows_html += f"<tr>{cells}</tr>"

    return f"<table>{header}{rows_html}</table>"


def _risk_contribution_html(contribs: List[RiskContribution]) -> str:
    if not contribs:
        return "<p>No strategy data.</p>"
    header = (
        "<tr><th>Strategy</th><th>Weight</th><th>Marginal</th>"
        "<th>% Contribution</th><th>Standalone Vol</th></tr>"
    )
    rows = ""
    for rc in contribs:
        rows += (
            f"<tr><td>{html_mod.escape(rc.strategy)}</td>"
            f"<td>{_fmt_pct(rc.weight)}</td>"
            f"<td>{_fmt_num(rc.marginal_contribution, 5)}</td>"
            f"<td>{_fmt_pct(rc.percent_contribution)}</td>"
            f"<td>{_fmt_pct(rc.standalone_vol)}</td></tr>"
        )
    return f"<table>{header}{rows}</table>"


def _drawdown_table_html(periods: List[DrawdownPeriod]) -> str:
    if not periods:
        return "<p>No drawdowns detected.</p>"
    header = (
        "<tr><th>#</th><th>Start</th><th>Trough</th><th>End</th>"
        "<th>Depth</th><th>Recovery (days)</th></tr>"
    )
    rows = ""
    for i, p in enumerate(periods, 1):
        end = str(p.end.date()) if p.end else "ongoing"
        rec = str(p.recovery_days) if p.recovery_days is not None else "N/A"
        rows += (
            f"<tr><td>{i}</td>"
            f"<td>{p.start.date()}</td><td>{p.trough.date()}</td>"
            f"<td>{end}</td>"
            f'<td class="{_color_class(p.depth)}">{_fmt_pct(p.depth)}</td>'
            f"<td>{rec}</td></tr>"
        )
    return f"<table>{header}{rows}</table>"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PortfolioAnalytics:
    """Comprehensive portfolio analytics engine.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns indexed by date.
    strategy_returns : dict, optional
        Mapping of strategy name to daily return series.
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.05 for 5%).
    """

    def __init__(
        self,
        returns: pd.Series,
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.returns = _to_series(returns)
        self.strategy_returns = strategy_returns or {}
        self.risk_free_rate = risk_free_rate

    # -- Metrics ----------------------------------------------------------

    def metrics(self) -> PortfolioMetrics:
        return build_portfolio_metrics(self.returns, self.risk_free_rate)

    # -- Rolling ----------------------------------------------------------

    def rolling(
        self, benchmark: Optional[pd.Series] = None,
    ) -> RollingResult:
        b = _to_series(benchmark) if benchmark is not None else None
        return compute_rolling(self.returns, b, self.risk_free_rate)

    # -- Benchmark --------------------------------------------------------

    def compare(
        self,
        benchmarks: Dict[str, pd.Series],
    ) -> List[BenchmarkComparison]:
        results = []
        for name, bench in benchmarks.items():
            results.append(compare_benchmark(
                self.returns, _to_series(bench), name, self.risk_free_rate
            ))
        return results

    # -- Risk contribution ------------------------------------------------

    def risk_contributions(self) -> List[RiskContribution]:
        return compute_risk_contributions(self.strategy_returns)

    # -- Return tables ----------------------------------------------------

    def return_tables(self) -> ReturnTable:
        return build_return_tables(self.returns)

    # -- Drawdowns --------------------------------------------------------

    def drawdowns(self) -> List[DrawdownPeriod]:
        return detect_drawdowns(self.returns)

    def drawdown_series(self) -> pd.Series:
        return compute_drawdown_series(self.returns)

    # -- Equity curve -----------------------------------------------------

    def equity_curve(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    # -- Report -----------------------------------------------------------

    def generate_report(
        self,
        output_path: Optional[str | Path] = None,
        benchmarks: Optional[Dict[str, pd.Series]] = None,
    ) -> Path:
        """Generate a self-contained dark-theme HTML report.

        Parameters
        ----------
        output_path : path, optional
            Where to save the HTML file. Defaults to
            ``./portfolio_report.html``.
        benchmarks : dict, optional
            Named benchmark return series for comparison.

        Returns
        -------
        Path to the generated HTML file.
        """
        if output_path is None:
            output_path = Path("portfolio_report.html")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.metrics()
        dd_periods = self.drawdowns()
        dd_series = self.drawdown_series()
        eq_curve = self.equity_curve()
        ret_tables = self.return_tables()
        risk_contribs = self.risk_contributions()

        comparisons: List[BenchmarkComparison] = []
        if benchmarks:
            comparisons = self.compare(benchmarks)

        # -- Build SVGs ---------------------------------------------------
        eq_svg = _svg_line_chart(
            {"Portfolio": eq_curve},
            title="Equity Curve",
        )
        dd_svg = _svg_line_chart(
            {"Drawdown": dd_series},
            title="Drawdown Timeline",
        )

        # -- Assemble HTML ------------------------------------------------
        sections = []

        # Metrics
        sections.append(
            '<div class="section"><h2>Portfolio Metrics</h2>'
            + _metrics_cards_html(metrics)
            + "</div>"
        )

        # Equity curve
        sections.append(
            '<div class="section"><h2>Equity Curve</h2>'
            + eq_svg
            + "</div>"
        )

        # Drawdown
        sections.append(
            '<div class="section"><h2>Drawdown</h2>'
            + dd_svg
            + "<h3>Drawdown Periods</h3>"
            + _drawdown_table_html(dd_periods)
            + "</div>"
        )

        # Benchmark comparison
        if comparisons:
            sections.append(
                '<div class="section"><h2>Benchmark Comparison</h2>'
                + _benchmark_table_html(comparisons)
                + "</div>"
            )

        # Risk contributions
        if risk_contribs:
            sections.append(
                '<div class="section"><h2>Risk Contribution by Strategy</h2>'
                + _risk_contribution_html(risk_contribs)
                + "</div>"
            )

        # Return tables
        sections.append(
            '<div class="section"><h2>Monthly Returns</h2>'
            + _df_to_html(ret_tables.monthly)
            + "</div>"
        )
        sections.append(
            '<div class="section"><h2>Quarterly Returns</h2>'
            + _df_to_html(ret_tables.quarterly)
            + "</div>"
        )
        sections.append(
            '<div class="section"><h2>Annual Returns</h2>'
            + _df_to_html(ret_tables.annual.to_frame("Return"))
            + "</div>"
        )

        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Portfolio Analytics Report</title>"
            f"<style>{_CSS}</style></head><body>"
            "<h1>Portfolio Analytics Report</h1>"
            + "".join(sections)
            + "</body></html>"
        )

        output_path.write_text(html, encoding="utf-8")
        return output_path
