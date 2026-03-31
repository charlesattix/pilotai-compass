"""
compass/strategy_report.py — Investor-quality strategy report generator.

Produces comprehensive HTML reports for any experiment with professional
styling, benchmark comparisons, and configurable templates.

Provides:
  1. Performance metrics (returns, Sharpe, Sortino, Calmar, max DD, win rate)
  2. Equity curve, drawdown chart, monthly returns heatmap
  3. Regime performance breakdown
  4. Risk metrics (VaR, CVaR) and rolling Sharpe plot
  5. Trade statistics (avg win/loss, profit factor, avg hold time)
  6. Benchmark comparison (vs SPY buy-and-hold, vs risk-free)
  7. Professional dark/light themes, configurable templates
  8. Batch mode for generating reports across all experiments

Usage:
    from compass.strategy_report import StrategyReportGenerator

    gen = StrategyReportGenerator(
        experiment_id="EXP-400",
        daily_returns=returns,
        spy_returns=spy,
        trades=trades_list,
    )
    html = gen.generate()

    # Batch mode:
    reports = generate_batch(experiments_dict, spy_returns=spy)
"""

from __future__ import annotations

import calendar
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR = 252
MONTHS_ABBR = [calendar.month_abbr[i] for i in range(1, 13)]

DEFAULT_SECTIONS = (
    "executive_summary",
    "performance_metrics",
    "equity_curve",
    "drawdown_chart",
    "monthly_heatmap",
    "rolling_sharpe",
    "regime_breakdown",
    "risk_metrics",
    "trade_statistics",
    "benchmark_comparison",
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Minimal trade record for report statistics."""
    pnl: float
    is_winner: bool
    hold_days: float = 0.0
    regime: str = ""


@dataclass
class PerformanceMetrics:
    """Computed performance metrics for a strategy."""
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float  # as positive fraction (e.g. 0.15 = 15%)
    max_drawdown_duration_days: int
    win_rate: float  # 0-1
    n_periods: int
    skewness: float
    kurtosis: float
    best_day: float
    worst_day: float


@dataclass
class RiskMetrics:
    """VaR and CVaR risk measures."""
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float


@dataclass
class TradeStatistics:
    """Aggregate trade-level statistics."""
    n_trades: int
    n_winners: int
    n_losers: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_hold_days: float
    expectancy: float  # avg_win * win_rate + avg_loss * (1 - win_rate)


@dataclass
class BenchmarkComparison:
    """Strategy vs benchmark comparison."""
    strategy_total_return: float
    strategy_sharpe: float
    spy_total_return: float
    spy_sharpe: float
    risk_free_total_return: float
    alpha: float  # strategy annual return - spy annual return
    beta: float  # regression beta to SPY
    information_ratio: float
    tracking_error: float


@dataclass
class RegimePerformance:
    """Performance metrics for a single regime."""
    regime: str
    n_days: int
    mean_daily_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    win_rate_daily: float


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    theme: str = "light"  # "light" or "dark"
    sections: Sequence[str] = DEFAULT_SECTIONS
    title: str = ""
    subtitle: str = ""
    rolling_window: int = 63  # ~3 months for rolling Sharpe
    risk_free_rate: float = 0.045
    var_confidence: Tuple[float, float] = (0.95, 0.99)
    include_footer: bool = True


@dataclass
class ReportData:
    """All computed data for the report."""
    experiment_id: str
    config: ReportConfig
    performance: PerformanceMetrics
    risk: RiskMetrics
    trades: Optional[TradeStatistics]
    benchmark: Optional[BenchmarkComparison]
    regimes: List[RegimePerformance]
    equity_curve: np.ndarray
    drawdown_series: np.ndarray
    rolling_sharpe: np.ndarray
    monthly_returns: Dict[Tuple[int, int], float]  # (year, month) -> return
    daily_returns: np.ndarray


# ── Metric computation ────────────────────────────────────────────────────────

def _compute_equity_curve(returns: np.ndarray, starting_value: float = 1.0) -> np.ndarray:
    """Cumulative equity curve from daily returns."""
    return starting_value * np.cumprod(1 + returns)


def _compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Drawdown series: (equity - running_max) / running_max."""
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.where(running_max > 0, running_max, 1.0)
    return dd


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction."""
    dd = _compute_drawdown_series(equity)
    return float(-np.min(dd)) if len(dd) > 0 else 0.0


def _max_drawdown_duration(equity: np.ndarray) -> int:
    """Longest drawdown duration in periods."""
    running_max = np.maximum.accumulate(equity)
    in_dd = equity < running_max
    if not np.any(in_dd):
        return 0
    max_dur = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def _sharpe_ratio(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - rf_daily
    std = np.std(excess)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(PERIODS_PER_YEAR))


def _sortino_ratio(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 1:
        return 0.0
    down_std = np.sqrt(np.mean(downside ** 2))
    if down_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / down_std * math.sqrt(PERIODS_PER_YEAR))


def _calmar_ratio(returns: np.ndarray) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_ret = float(np.mean(returns) * PERIODS_PER_YEAR)
    equity = _compute_equity_curve(returns)
    mdd = _max_drawdown(equity)
    if mdd < 1e-12:
        return 0.0
    return ann_ret / mdd


def _var_cvar(returns: np.ndarray, confidence: float) -> Tuple[float, float]:
    """Historical VaR and CVaR (positive loss numbers)."""
    alpha = 1.0 - confidence
    var = -float(np.percentile(returns, alpha * 100))
    tail = returns[returns <= -var]
    cvar = -float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


def _monthly_returns(returns: np.ndarray, start_year: int = 2020, start_month: int = 1) -> Dict[Tuple[int, int], float]:
    """Aggregate daily returns into monthly returns.

    Uses sequential month assignment: first ~21 days = month 1, etc.
    If start_year/start_month provided, labels months accordingly.
    """
    result: Dict[Tuple[int, int], float] = {}
    idx = 0
    year, month = start_year, start_month

    while idx < len(returns):
        # ~21 trading days per month
        end = min(idx + 21, len(returns))
        chunk = returns[idx:end]
        monthly_ret = float(np.prod(1 + chunk) - 1)
        result[(year, month)] = monthly_ret
        idx = end
        month += 1
        if month > 12:
            month = 1
            year += 1

    return result


def _rolling_sharpe(returns: np.ndarray, window: int, rf_daily: float = 0.0) -> np.ndarray:
    """Rolling annualized Sharpe ratio."""
    n = len(returns)
    if n < window:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = returns[i - window + 1:i + 1]
        excess = chunk - rf_daily
        std = np.std(excess)
        if std > 1e-12:
            result[i] = float(np.mean(excess) / std * math.sqrt(PERIODS_PER_YEAR))
        else:
            result[i] = 0.0
    return result


def compute_performance(
    returns: np.ndarray,
    risk_free_rate: float = 0.045,
) -> PerformanceMetrics:
    """Compute all performance metrics from daily returns."""
    n = len(returns)
    rf_daily = risk_free_rate / PERIODS_PER_YEAR

    equity = _compute_equity_curve(returns)
    total_return = float(equity[-1] / equity[0] - 1) if n > 0 else 0.0
    ann_return = float(np.mean(returns) * PERIODS_PER_YEAR)
    ann_vol = float(np.std(returns) * math.sqrt(PERIODS_PER_YEAR))
    sharpe = _sharpe_ratio(returns, rf_daily)
    sortino = _sortino_ratio(returns, rf_daily)
    calmar = _calmar_ratio(returns)
    mdd = _max_drawdown(equity)
    mdd_dur = _max_drawdown_duration(equity)
    win_rate = float(np.mean(returns > 0)) if n > 0 else 0.0

    return PerformanceMetrics(
        total_return=round(total_return, 6),
        annual_return=round(ann_return, 6),
        annual_vol=round(ann_vol, 6),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        calmar=round(calmar, 4),
        max_drawdown=round(mdd, 6),
        max_drawdown_duration_days=mdd_dur,
        win_rate=round(win_rate, 4),
        n_periods=n,
        skewness=round(float(_skew(returns)), 4),
        kurtosis=round(float(_kurtosis(returns)), 4),
        best_day=round(float(np.max(returns)), 6) if n > 0 else 0.0,
        worst_day=round(float(np.min(returns)), 6) if n > 0 else 0.0,
    )


def _skew(x: np.ndarray) -> float:
    """Sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def compute_risk(returns: np.ndarray) -> RiskMetrics:
    """Compute VaR and CVaR at 95% and 99%."""
    v95, c95 = _var_cvar(returns, 0.95)
    v99, c99 = _var_cvar(returns, 0.99)
    return RiskMetrics(
        var_95=round(v95, 6),
        cvar_95=round(c95, 6),
        var_99=round(v99, 6),
        cvar_99=round(c99, 6),
    )


def compute_trade_statistics(trades: List[TradeRecord]) -> TradeStatistics:
    """Compute aggregate trade statistics."""
    if not trades:
        return TradeStatistics(
            n_trades=0, n_winners=0, n_losers=0, win_rate=0.0,
            avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
            profit_factor=0.0, avg_hold_days=0.0, expectancy=0.0,
        )

    pnls = np.array([t.pnl for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    n_winners = len(winners)
    n_losers = len(losers)
    n_trades = len(trades)
    win_rate = n_winners / n_trades if n_trades > 0 else 0.0

    avg_win = float(np.mean(winners)) if n_winners > 0 else 0.0
    avg_loss = float(np.mean(losers)) if n_losers > 0 else 0.0
    largest_win = float(np.max(pnls)) if n_trades > 0 else 0.0
    largest_loss = float(np.min(pnls)) if n_trades > 0 else 0.0

    gross_profit = float(np.sum(winners)) if n_winners > 0 else 0.0
    gross_loss = abs(float(np.sum(losers))) if n_losers > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

    return TradeStatistics(
        n_trades=n_trades,
        n_winners=n_winners,
        n_losers=n_losers,
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        profit_factor=round(profit_factor, 4),
        avg_hold_days=round(avg_hold, 1),
        expectancy=round(expectancy, 2),
    )


def compute_benchmark(
    returns: np.ndarray,
    spy_returns: np.ndarray,
    risk_free_rate: float = 0.045,
) -> BenchmarkComparison:
    """Compare strategy returns to SPY and risk-free benchmarks."""
    rf_daily = risk_free_rate / PERIODS_PER_YEAR
    n = len(returns)

    strat_total = float(np.prod(1 + returns) - 1)
    spy_total = float(np.prod(1 + spy_returns) - 1)
    rf_total = float((1 + rf_daily) ** n - 1)

    strat_sharpe = _sharpe_ratio(returns, rf_daily)
    spy_sharpe = _sharpe_ratio(spy_returns, rf_daily)

    strat_ann = float(np.mean(returns) * PERIODS_PER_YEAR)
    spy_ann = float(np.mean(spy_returns) * PERIODS_PER_YEAR)
    alpha = strat_ann - spy_ann

    # Beta via OLS: returns ~ alpha + beta * spy
    spy_var = np.var(spy_returns)
    if spy_var > 1e-16:
        beta = float(np.cov(returns, spy_returns)[0, 1] / spy_var)
    else:
        beta = 0.0

    # Information ratio = alpha / tracking_error
    active_returns = returns - spy_returns
    tracking_error = float(np.std(active_returns) * math.sqrt(PERIODS_PER_YEAR))
    ir = alpha / tracking_error if tracking_error > 1e-12 else 0.0

    return BenchmarkComparison(
        strategy_total_return=round(strat_total, 6),
        strategy_sharpe=round(strat_sharpe, 4),
        spy_total_return=round(spy_total, 6),
        spy_sharpe=round(spy_sharpe, 4),
        risk_free_total_return=round(rf_total, 6),
        alpha=round(alpha, 6),
        beta=round(beta, 4),
        information_ratio=round(ir, 4),
        tracking_error=round(tracking_error, 6),
    )


def compute_regime_performance(
    returns: np.ndarray,
    regimes: np.ndarray,
    regime_labels: Optional[List[str]] = None,
) -> List[RegimePerformance]:
    """Compute performance per regime.

    Args:
        returns: Daily returns array.
        regimes: Integer array of regime labels (same length as returns).
        regime_labels: Optional list mapping regime int to label string.
    """
    unique = sorted(set(regimes))
    if regime_labels is None:
        regime_labels = [f"Regime_{i}" for i in range(max(unique) + 1)]

    results = []
    for r in unique:
        mask = regimes == r
        r_ret = returns[mask]
        n_days = int(np.sum(mask))
        if n_days < 2:
            continue
        label = regime_labels[r] if r < len(regime_labels) else f"Regime_{r}"
        ann_ret = float(np.mean(r_ret) * PERIODS_PER_YEAR)
        ann_vol = float(np.std(r_ret) * math.sqrt(PERIODS_PER_YEAR))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        eq = _compute_equity_curve(r_ret)
        mdd = _max_drawdown(eq)
        wr = float(np.mean(r_ret > 0))

        results.append(RegimePerformance(
            regime=label,
            n_days=n_days,
            mean_daily_return=round(float(np.mean(r_ret)), 6),
            annual_return=round(ann_ret, 6),
            annual_vol=round(ann_vol, 6),
            sharpe=round(sharpe, 4),
            max_drawdown=round(mdd, 6),
            win_rate_daily=round(wr, 4),
        ))

    return results


# ── Theme CSS ─────────────────────────────────────────────────────────────────

_LIGHT_THEME = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }
  h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }
  h2 { color: #334155; margin-top: 2em; }
  .meta { color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }
  .kpi-row { display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }
  .kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
         padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }
  .kpi .value { font-size: 1.5em; font-weight: 700; }
  .kpi .label { font-size: 0.75em; color: #64748b; margin-top: 0.2em; }
  .good { color: #16a34a; }
  .bad { color: #dc2626; }
  .warn { color: #d97706; }
  .neutral { color: #64748b; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }
  th { background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }
  td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }
  .chart { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
           padding: 1.5em; margin: 1.5em 0; overflow-x: auto; }
  .section { margin-bottom: 2.5em; }
  .heatmap-cell { padding: 4px 8px; text-align: center; font-size: 0.8em;
                  border: 1px solid #e2e8f0; min-width: 50px; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
           color: #94a3b8; font-size: 0.8em; }
"""

_DARK_THEME = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #0f172a; color: #e2e8f0; }
  h1 { color: #f1f5f9; border-bottom: 2px solid #334155; padding-bottom: 0.4em; }
  h2 { color: #cbd5e1; margin-top: 2em; }
  .meta { color: #94a3b8; font-size: 0.9em; margin-bottom: 1.5em; }
  .kpi-row { display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }
  .kpi { background: #1e293b; border: 1px solid #334155; border-radius: 8px;
         padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }
  .kpi .value { font-size: 1.5em; font-weight: 700; }
  .kpi .label { font-size: 0.75em; color: #94a3b8; margin-top: 0.2em; }
  .good { color: #4ade80; }
  .bad { color: #f87171; }
  .warn { color: #fbbf24; }
  .neutral { color: #94a3b8; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }
  th { background: #1e293b; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #475569; font-weight: 600; color: #e2e8f0; }
  td { padding: 6px 10px; border-bottom: 1px solid #334155; }
  .chart { background: #1e293b; border: 1px solid #334155; border-radius: 8px;
           padding: 1.5em; margin: 1.5em 0; overflow-x: auto; }
  .section { margin-bottom: 2.5em; }
  .heatmap-cell { padding: 4px 8px; text-align: center; font-size: 0.8em;
                  border: 1px solid #334155; min-width: 50px; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #334155;
           color: #64748b; font-size: 0.8em; }
"""


# ── Report generator ──────────────────────────────────────────────────────────

class StrategyReportGenerator:
    """Investor-quality strategy report generator.

    Args:
        experiment_id: Experiment identifier (e.g. "EXP-400").
        daily_returns: numpy array of strategy daily returns.
        spy_returns: Optional SPY daily returns for benchmark comparison.
        trades: Optional list of TradeRecord for trade statistics.
        regimes: Optional integer array of regime classifications.
        regime_labels: Optional list mapping regime int to name.
        config: Report configuration (theme, sections, etc.).
        start_year: Year label for monthly returns (default 2024).
        start_month: Month label for monthly returns (default 1).
    """

    def __init__(
        self,
        experiment_id: str,
        daily_returns: np.ndarray,
        spy_returns: Optional[np.ndarray] = None,
        trades: Optional[List[TradeRecord]] = None,
        regimes: Optional[np.ndarray] = None,
        regime_labels: Optional[List[str]] = None,
        config: Optional[ReportConfig] = None,
        start_year: int = 2024,
        start_month: int = 1,
    ):
        self.experiment_id = experiment_id
        self.returns = np.asarray(daily_returns, dtype=float)
        self.n_periods = len(self.returns)

        if self.n_periods < 2:
            raise ValueError("Need at least 2 return periods to generate report")

        self.spy_returns = (
            np.asarray(spy_returns, dtype=float) if spy_returns is not None else None
        )
        if self.spy_returns is not None and len(self.spy_returns) != self.n_periods:
            raise ValueError(
                f"spy_returns length ({len(self.spy_returns)}) != daily_returns ({self.n_periods})"
            )

        self.trades = trades
        self.regimes = np.asarray(regimes, dtype=int) if regimes is not None else None
        self.regime_labels = regime_labels
        self.config = config or ReportConfig()
        self.start_year = start_year
        self.start_month = start_month

    def compute_all(self) -> ReportData:
        """Compute all data needed for the report."""
        rf = self.config.risk_free_rate
        perf = compute_performance(self.returns, rf)
        risk = compute_risk(self.returns)

        trade_stats = compute_trade_statistics(self.trades) if self.trades else None

        benchmark = None
        if self.spy_returns is not None:
            benchmark = compute_benchmark(self.returns, self.spy_returns, rf)

        regime_perfs: List[RegimePerformance] = []
        if self.regimes is not None and len(self.regimes) == self.n_periods:
            regime_perfs = compute_regime_performance(
                self.returns, self.regimes, self.regime_labels
            )

        equity = _compute_equity_curve(self.returns)
        dd = _compute_drawdown_series(equity)
        rf_daily = rf / PERIODS_PER_YEAR
        roll_sharpe = _rolling_sharpe(self.returns, self.config.rolling_window, rf_daily)
        monthly = _monthly_returns(self.returns, self.start_year, self.start_month)

        return ReportData(
            experiment_id=self.experiment_id,
            config=self.config,
            performance=perf,
            risk=risk,
            trades=trade_stats,
            benchmark=benchmark,
            regimes=regime_perfs,
            equity_curve=equity,
            drawdown_series=dd,
            rolling_sharpe=roll_sharpe,
            monthly_returns=monthly,
            daily_returns=self.returns,
        )

    def generate(self) -> str:
        """Generate the full HTML report."""
        data = self.compute_all()
        return render_html(data)


# ── HTML rendering ────────────────────────────────────────────────────────────

def render_html(data: ReportData) -> str:
    """Render ReportData into a self-contained HTML document."""
    cfg = data.config
    theme_css = _DARK_THEME if cfg.theme == "dark" else _LIGHT_THEME
    title = cfg.title or f"Strategy Report — {data.experiment_id}"
    subtitle = cfg.subtitle or ""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections_html = ""

    for section in cfg.sections:
        renderer = _SECTION_RENDERERS.get(section)
        if renderer:
            sections_html += renderer(data)

    footer = ""
    if cfg.include_footer:
        footer = (
            f'<footer>Generated by COMPASS Strategy Report Engine | '
            f'{data.experiment_id} | {now}</footer>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{theme_css}</style>
</head>
<body>
<h1>{title}</h1>
{"<p class='meta'>" + subtitle + "</p>" if subtitle else ""}
<div class="meta">Generated: {now} | {data.performance.n_periods} trading days</div>
{sections_html}
{footer}
</body>
</html>"""


def _pnl_cls(val: float) -> str:
    """CSS class for positive/negative values."""
    if val > 0.001:
        return "good"
    elif val < -0.001:
        return "bad"
    return "neutral"


def _fmt_pct(val: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def _fmt_dollar(val: float) -> str:
    """Format a dollar amount."""
    return f"${val:,.2f}"


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_executive_summary(data: ReportData) -> str:
    p = data.performance
    return f"""
<div class="section">
<h2>Executive Summary</h2>
<div class="kpi-row">
  <div class="kpi"><div class="value {_pnl_cls(p.total_return)}">{_fmt_pct(p.total_return)}</div><div class="label">Total Return</div></div>
  <div class="kpi"><div class="value {_pnl_cls(p.sharpe)}">{p.sharpe:.2f}</div><div class="label">Sharpe Ratio</div></div>
  <div class="kpi"><div class="value bad">{_fmt_pct(p.max_drawdown)}</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value {_pnl_cls(p.sortino)}">{p.sortino:.2f}</div><div class="label">Sortino Ratio</div></div>
  <div class="kpi"><div class="value">{_fmt_pct(p.win_rate)}</div><div class="label">Win Rate (Daily)</div></div>
  <div class="kpi"><div class="value">{_fmt_pct(p.annual_vol)}</div><div class="label">Annual Vol</div></div>
</div>
</div>"""


def _render_performance_metrics(data: ReportData) -> str:
    p = data.performance
    rows = f"""
    <tr><td>Total Return</td><td class="{_pnl_cls(p.total_return)}">{_fmt_pct(p.total_return)}</td></tr>
    <tr><td>Annual Return</td><td class="{_pnl_cls(p.annual_return)}">{_fmt_pct(p.annual_return)}</td></tr>
    <tr><td>Annual Volatility</td><td>{_fmt_pct(p.annual_vol)}</td></tr>
    <tr><td>Sharpe Ratio</td><td class="{_pnl_cls(p.sharpe)}">{p.sharpe:.3f}</td></tr>
    <tr><td>Sortino Ratio</td><td class="{_pnl_cls(p.sortino)}">{p.sortino:.3f}</td></tr>
    <tr><td>Calmar Ratio</td><td class="{_pnl_cls(p.calmar)}">{p.calmar:.3f}</td></tr>
    <tr><td>Max Drawdown</td><td class="bad">{_fmt_pct(p.max_drawdown)}</td></tr>
    <tr><td>Max DD Duration</td><td>{p.max_drawdown_duration_days} days</td></tr>
    <tr><td>Win Rate (Daily)</td><td>{_fmt_pct(p.win_rate)}</td></tr>
    <tr><td>Best Day</td><td class="good">{_fmt_pct(p.best_day, 3)}</td></tr>
    <tr><td>Worst Day</td><td class="bad">{_fmt_pct(p.worst_day, 3)}</td></tr>
    <tr><td>Skewness</td><td>{p.skewness:.3f}</td></tr>
    <tr><td>Excess Kurtosis</td><td>{p.kurtosis:.3f}</td></tr>
    """
    return f"""
<div class="section">
<h2>Performance Metrics</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{rows}</tbody></table>
</div>"""


def _render_equity_curve(data: ReportData) -> str:
    svg = _svg_line_chart(
        data.equity_curve.tolist(),
        width=700, height=250,
        color="#3b82f6",
        y_label="Equity",
        fill=True,
    )
    return f"""
<div class="section">
<h2>Equity Curve</h2>
<div class="chart">{svg}</div>
</div>"""


def _render_drawdown_chart(data: ReportData) -> str:
    svg = _svg_line_chart(
        data.drawdown_series.tolist(),
        width=700, height=200,
        color="#dc2626",
        y_label="Drawdown",
        fill=True,
        is_negative=True,
    )
    return f"""
<div class="section">
<h2>Drawdown</h2>
<div class="chart">{svg}</div>
</div>"""


def _render_monthly_heatmap(data: ReportData) -> str:
    if not data.monthly_returns:
        return ""

    years = sorted(set(y for y, _ in data.monthly_returns.keys()))
    header = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in MONTHS_ABBR) + "<th>Annual</th></tr>"

    rows = ""
    for year in years:
        row = f"<tr><td><b>{year}</b></td>"
        annual = 1.0
        for month in range(1, 13):
            val = data.monthly_returns.get((year, month))
            if val is not None:
                annual *= (1 + val)
                bg = _heatmap_color(val, data.config.theme)
                row += f'<td class="heatmap-cell" style="background:{bg}">{val:+.1%}</td>'
            else:
                row += '<td class="heatmap-cell">—</td>'
        annual_ret = annual - 1.0
        bg = _heatmap_color(annual_ret, data.config.theme)
        row += f'<td class="heatmap-cell" style="background:{bg};font-weight:700">{annual_ret:+.1%}</td>'
        rows += row + "</tr>\n"

    return f"""
<div class="section">
<h2>Monthly Returns</h2>
<table style="font-size:0.82em">{header}{rows}</table>
</div>"""


def _heatmap_color(val: float, theme: str = "light") -> str:
    """Generate a heatmap background color for a return value."""
    intensity = min(abs(val) * 500, 100)
    if theme == "dark":
        if val > 0:
            return f"rgba(74,222,128,{intensity / 100 * 0.4:.2f})"
        elif val < 0:
            return f"rgba(248,113,113,{intensity / 100 * 0.4:.2f})"
        return "transparent"
    else:
        if val > 0:
            return f"rgba(22,163,74,{intensity / 100 * 0.3:.2f})"
        elif val < 0:
            return f"rgba(220,38,38,{intensity / 100 * 0.3:.2f})"
        return "transparent"


def _render_rolling_sharpe(data: ReportData) -> str:
    valid = data.rolling_sharpe[~np.isnan(data.rolling_sharpe)]
    if len(valid) < 2:
        return '<div class="section"><h2>Rolling Sharpe</h2><p>Insufficient data</p></div>'

    svg = _svg_line_chart(
        valid.tolist(),
        width=700, height=200,
        color="#8b5cf6",
        y_label="Sharpe",
        zero_line=True,
    )
    return f"""
<div class="section">
<h2>Rolling Sharpe ({data.config.rolling_window}-day)</h2>
<div class="chart">{svg}</div>
</div>"""


def _render_regime_breakdown(data: ReportData) -> str:
    if not data.regimes:
        return ""

    rows = ""
    for rp in data.regimes:
        rows += (
            f"<tr><td>{rp.regime}</td><td>{rp.n_days}</td>"
            f"<td class='{_pnl_cls(rp.annual_return)}'>{_fmt_pct(rp.annual_return)}</td>"
            f"<td>{_fmt_pct(rp.annual_vol)}</td>"
            f"<td class='{_pnl_cls(rp.sharpe)}'>{rp.sharpe:.2f}</td>"
            f"<td class='bad'>{_fmt_pct(rp.max_drawdown)}</td>"
            f"<td>{_fmt_pct(rp.win_rate_daily)}</td></tr>\n"
        )

    return f"""
<div class="section">
<h2>Regime Breakdown</h2>
<table>
<thead><tr><th>Regime</th><th>Days</th><th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""


def _render_risk_metrics(data: ReportData) -> str:
    r = data.risk
    return f"""
<div class="section">
<h2>Risk Metrics</h2>
<table>
<thead><tr><th>Measure</th><th>95% Confidence</th><th>99% Confidence</th></tr></thead>
<tbody>
<tr><td>Value-at-Risk (daily)</td><td class="bad">{_fmt_pct(r.var_95, 3)}</td><td class="bad">{_fmt_pct(r.var_99, 3)}</td></tr>
<tr><td>CVaR / Expected Shortfall</td><td class="bad">{_fmt_pct(r.cvar_95, 3)}</td><td class="bad">{_fmt_pct(r.cvar_99, 3)}</td></tr>
</tbody>
</table>
</div>"""


def _render_trade_statistics(data: ReportData) -> str:
    ts = data.trades
    if ts is None:
        return ""

    return f"""
<div class="section">
<h2>Trade Statistics</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Total Trades</td><td>{ts.n_trades}</td></tr>
<tr><td>Winners / Losers</td><td>{ts.n_winners} / {ts.n_losers}</td></tr>
<tr><td>Win Rate</td><td>{_fmt_pct(ts.win_rate)}</td></tr>
<tr><td>Average Win</td><td class="good">{_fmt_dollar(ts.avg_win)}</td></tr>
<tr><td>Average Loss</td><td class="bad">{_fmt_dollar(ts.avg_loss)}</td></tr>
<tr><td>Largest Win</td><td class="good">{_fmt_dollar(ts.largest_win)}</td></tr>
<tr><td>Largest Loss</td><td class="bad">{_fmt_dollar(ts.largest_loss)}</td></tr>
<tr><td>Profit Factor</td><td class="{_pnl_cls(ts.profit_factor - 1)}">{ts.profit_factor:.2f}</td></tr>
<tr><td>Avg Hold Time</td><td>{ts.avg_hold_days:.1f} days</td></tr>
<tr><td>Expectancy</td><td class="{_pnl_cls(ts.expectancy)}">{_fmt_dollar(ts.expectancy)}</td></tr>
</tbody>
</table>
</div>"""


def _render_benchmark_comparison(data: ReportData) -> str:
    bm = data.benchmark
    if bm is None:
        return ""

    return f"""
<div class="section">
<h2>Benchmark Comparison</h2>
<table>
<thead><tr><th>Metric</th><th>Strategy</th><th>SPY B&amp;H</th><th>Risk-Free</th></tr></thead>
<tbody>
<tr><td>Total Return</td>
    <td class="{_pnl_cls(bm.strategy_total_return)}">{_fmt_pct(bm.strategy_total_return)}</td>
    <td class="{_pnl_cls(bm.spy_total_return)}">{_fmt_pct(bm.spy_total_return)}</td>
    <td>{_fmt_pct(bm.risk_free_total_return)}</td></tr>
<tr><td>Sharpe Ratio</td>
    <td class="{_pnl_cls(bm.strategy_sharpe)}">{bm.strategy_sharpe:.3f}</td>
    <td class="{_pnl_cls(bm.spy_sharpe)}">{bm.spy_sharpe:.3f}</td>
    <td>—</td></tr>
<tr><td>Alpha (ann.)</td><td class="{_pnl_cls(bm.alpha)}" colspan="3">{_fmt_pct(bm.alpha)}</td></tr>
<tr><td>Beta (to SPY)</td><td colspan="3">{bm.beta:.3f}</td></tr>
<tr><td>Information Ratio</td><td class="{_pnl_cls(bm.information_ratio)}" colspan="3">{bm.information_ratio:.3f}</td></tr>
<tr><td>Tracking Error</td><td colspan="3">{_fmt_pct(bm.tracking_error)}</td></tr>
</tbody>
</table>
</div>"""


# Map section names to rendering functions
_SECTION_RENDERERS: Dict[str, Any] = {
    "executive_summary": _render_executive_summary,
    "performance_metrics": _render_performance_metrics,
    "equity_curve": _render_equity_curve,
    "drawdown_chart": _render_drawdown_chart,
    "monthly_heatmap": _render_monthly_heatmap,
    "rolling_sharpe": _render_rolling_sharpe,
    "regime_breakdown": _render_regime_breakdown,
    "risk_metrics": _render_risk_metrics,
    "trade_statistics": _render_trade_statistics,
    "benchmark_comparison": _render_benchmark_comparison,
}


# ── SVG helpers ───────────────────────────────────────────────────────────────

def _svg_line_chart(
    values: List[float],
    width: int = 600,
    height: int = 200,
    color: str = "#3b82f6",
    y_label: str = "",
    fill: bool = False,
    is_negative: bool = False,
    zero_line: bool = False,
) -> str:
    """Render a simple SVG line chart."""
    n = len(values)
    if n < 2:
        return "<p>Insufficient data</p>"

    pad_l, pad_r, pad_t, pad_b = 55, 15, 15, 25
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1
    y_range = y_max - y_min

    points = []
    for i, v in enumerate(values):
        x = pad_l + (i / (n - 1)) * plot_w
        y = pad_t + (1 - (v - y_min) / y_range) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>'

    fill_svg = ""
    if fill:
        baseline_y = pad_t + plot_h if not is_negative else pad_t
        fill_points = f"{pad_l},{baseline_y} " + " ".join(points) + f" {pad_l + plot_w},{baseline_y}"
        fill_svg = f'<polygon points="{fill_points}" fill="{color}" fill-opacity="0.1"/>'

    zero_svg = ""
    if zero_line and y_min < 0 < y_max:
        zy = pad_t + (1 - (0 - y_min) / y_range) * plot_h
        zero_svg = (
            f'<line x1="{pad_l}" y1="{zy:.1f}" x2="{width - pad_r}" y2="{zy:.1f}" '
            f'stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="4,3"/>'
        )

    # Y-axis labels (5 ticks)
    y_labels = ""
    for i in range(5):
        frac = i / 4
        val = y_min + frac * y_range
        y = pad_t + (1 - frac) * plot_h
        if abs(val) < 1:
            label = f"{val:.2%}" if abs(y_range) < 1 else f"{val:.2f}"
        else:
            label = f"{val:.1f}"
        y_labels += (
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#94a3b8">{label}</text>\n'
        )

    # Y-axis title
    y_title = ""
    if y_label:
        y_title = (
            f'<text x="10" y="{pad_t + plot_h / 2}" text-anchor="middle" '
            f'font-size="10" fill="#64748b" transform="rotate(-90,10,{pad_t + plot_h / 2})">'
            f'{y_label}</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        f'{y_labels}{y_title}{zero_svg}{fill_svg}{polyline}</svg>'
    )


# ── Batch mode ────────────────────────────────────────────────────────────────

def generate_batch(
    experiments: Dict[str, np.ndarray],
    spy_returns: Optional[np.ndarray] = None,
    trades: Optional[Dict[str, List[TradeRecord]]] = None,
    regimes: Optional[np.ndarray] = None,
    regime_labels: Optional[List[str]] = None,
    config: Optional[ReportConfig] = None,
    output_dir: str = "reports",
    start_year: int = 2024,
    start_month: int = 1,
) -> Dict[str, str]:
    """Generate reports for all experiments at once.

    Args:
        experiments: Dict mapping experiment ID to daily returns array.
        spy_returns: Optional SPY returns (shared across experiments).
        trades: Optional dict mapping experiment ID to trade records.
        regimes: Optional shared regime array.
        regime_labels: Optional regime label list.
        config: Shared report configuration.
        output_dir: Directory to write HTML files.
        start_year: Year label for monthly returns.
        start_month: Month label for monthly returns.

    Returns:
        Dict mapping experiment ID to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    for eid, returns in sorted(experiments.items()):
        exp_trades = trades.get(eid) if trades else None
        gen = StrategyReportGenerator(
            experiment_id=eid,
            daily_returns=returns,
            spy_returns=spy_returns,
            trades=exp_trades,
            regimes=regimes,
            regime_labels=regime_labels,
            config=config,
            start_year=start_year,
            start_month=start_month,
        )
        html = gen.generate()
        path = os.path.join(output_dir, f"strategy_{eid}.html")
        with open(path, "w") as f:
            f.write(html)
        paths[eid] = path
        logger.info("Strategy report written: %s", path)

    logger.info("Batch complete: %d reports generated in %s", len(paths), output_dir)
    return paths
