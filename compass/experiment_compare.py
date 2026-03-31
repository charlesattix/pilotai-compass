"""Experiment comparison module for comparing multiple trading experiments."""

from __future__ import annotations

import math
import io
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExperimentMetrics:
    """Per-experiment performance metrics."""

    experiment_id: str
    sharpe: float
    sortino: float
    calmar: float
    total_return: float
    annual_return: float
    max_dd: float
    win_rate: float
    avg_trade_duration: float
    profit_factor: float


@dataclass
class PairComparison:
    """Statistical comparison between two experiments."""

    exp_a: str
    exp_b: str
    mean_diff: float
    t_stat: float
    p_value: float
    sharpe_diff: float
    ci_lower: float
    ci_upper: float


@dataclass
class CompareResult:
    """Full comparison result across all experiments."""

    experiment_metrics: list[ExperimentMetrics]
    pair_comparisons: list[PairComparison]
    correlation_matrix: pd.DataFrame
    best_experiment: str
    generated_at: str


class ExperimentComparer:
    """Compare multiple trading experiments by their daily return series."""

    TRADING_DAYS_PER_YEAR = 252
    BOOTSTRAP_SAMPLES = 1000

    def __init__(self, experiment_returns: dict[str, pd.Series]) -> None:
        self.experiment_returns = experiment_returns
        self._ids = sorted(experiment_returns.keys())

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def compare(self) -> CompareResult:
        """Run the full comparison and return a CompareResult."""
        metrics = [self._compute_metrics(eid) for eid in self._ids]
        pairs = self._compute_pair_comparisons()
        corr = self._compute_correlation_matrix()
        best = self._select_best(metrics)
        return CompareResult(
            experiment_metrics=metrics,
            pair_comparisons=pairs,
            correlation_matrix=corr,
            best_experiment=best,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def generate_report(self) -> str:
        """Generate an HTML report with tables, charts and heatmap."""
        result = self.compare()
        html_parts: list[str] = []
        html_parts.append(self._html_header())
        html_parts.append(self._metrics_table_html(result))
        html_parts.append(self._equity_curve_svg(result))
        html_parts.append(self._drawdown_svg(result))
        html_parts.append(self._pair_comparisons_html(result))
        html_parts.append(self._correlation_heatmap_svg(result))
        html_parts.append(self._monthly_returns_html(result))
        html_parts.append(self._html_footer())
        return "\n".join(html_parts)

    # ------------------------------------------------------------------ #
    #  Per-experiment metrics                                             #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, eid: str) -> ExperimentMetrics:
        returns = self.experiment_returns[eid].dropna()
        n = len(returns)
        if n == 0:
            return ExperimentMetrics(
                experiment_id=eid,
                sharpe=0.0,
                sortino=0.0,
                calmar=0.0,
                total_return=0.0,
                annual_return=0.0,
                max_dd=0.0,
                win_rate=0.0,
                avg_trade_duration=0.0,
                profit_factor=0.0,
            )

        mean_r = float(np.mean(returns.values))
        std_r = float(np.std(returns.values, ddof=1)) if n > 1 else 0.0
        td = self.TRADING_DAYS_PER_YEAR

        # Sharpe
        sharpe = (mean_r / std_r * math.sqrt(td)) if std_r > 0 else 0.0

        # Sortino
        downside = returns.values[returns.values < 0]
        down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (mean_r / down_std * math.sqrt(td)) if down_std > 0 else 0.0

        # Equity & drawdown
        cum = (1 + returns).cumprod()
        total_return = float(cum.iloc[-1] - 1)
        annual_return = float((1 + total_return) ** (td / max(n, 1)) - 1)
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())  # negative number

        # Calmar
        calmar = (annual_return / abs(max_dd)) if max_dd != 0 else 0.0

        # Win rate
        win_rate = float(np.mean(returns.values > 0))

        # Avg trade duration (from sign changes)
        avg_trade_duration = self._avg_trade_duration(returns)

        # Profit factor
        gains = float(np.sum(returns.values[returns.values > 0]))
        losses = float(abs(np.sum(returns.values[returns.values < 0])))
        profit_factor = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

        return ExperimentMetrics(
            experiment_id=eid,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            total_return=total_return,
            annual_return=annual_return,
            max_dd=max_dd,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            profit_factor=profit_factor,
        )

    @staticmethod
    def _avg_trade_duration(returns: pd.Series) -> float:
        """Average duration between sign changes in the return series."""
        vals = returns.values
        if len(vals) < 2:
            return float(len(vals))
        signs = np.sign(vals)
        # Treat zero as continuation of previous sign
        changes = np.where(signs[1:] != signs[:-1])[0]
        if len(changes) == 0:
            return float(len(vals))
        durations: list[int] = []
        prev = 0
        for idx in changes:
            durations.append(int(idx + 1 - prev))
            prev = idx + 1
        durations.append(len(vals) - prev)
        return float(np.mean(durations))

    # ------------------------------------------------------------------ #
    #  Pair comparisons (t-test + bootstrap)                             #
    # ------------------------------------------------------------------ #

    def _compute_pair_comparisons(self) -> list[PairComparison]:
        pairs: list[PairComparison] = []
        for i, a in enumerate(self._ids):
            for b in self._ids[i + 1 :]:
                pairs.append(self._compare_pair(a, b))
        return pairs

    def _compare_pair(self, a: str, b: str) -> PairComparison:
        ra = self.experiment_returns[a].dropna()
        rb = self.experiment_returns[b].dropna()

        # Align on common index
        common = ra.index.intersection(rb.index)
        ra = ra.loc[common]
        rb = rb.loc[common]
        diff = ra.values - rb.values
        n = len(diff)

        if n < 2:
            return PairComparison(
                exp_a=a, exp_b=b, mean_diff=0.0,
                t_stat=0.0, p_value=1.0, sharpe_diff=0.0,
                ci_lower=0.0, ci_upper=0.0,
            )

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        # Manual paired t-test
        if std_diff > 0:
            t_stat = mean_diff / (std_diff / math.sqrt(n))
        else:
            t_stat = 0.0

        # p-value from normal approximation (two-tailed)
        p_value = self._normal_two_tail_p(t_stat)

        # Sharpe difference
        sharpe_a = self._quick_sharpe(ra.values)
        sharpe_b = self._quick_sharpe(rb.values)
        sharpe_diff = sharpe_a - sharpe_b

        # Bootstrap CI on Sharpe difference
        ci_lower, ci_upper = self._bootstrap_sharpe_ci(ra.values, rb.values)

        return PairComparison(
            exp_a=a, exp_b=b,
            mean_diff=mean_diff,
            t_stat=t_stat,
            p_value=p_value,
            sharpe_diff=sharpe_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def _quick_sharpe(self, r: np.ndarray) -> float:
        if len(r) < 2:
            return 0.0
        s = float(np.std(r, ddof=1))
        if s == 0:
            return 0.0
        return float(np.mean(r)) / s * math.sqrt(self.TRADING_DAYS_PER_YEAR)

    def _bootstrap_sharpe_ci(
        self, ra: np.ndarray, rb: np.ndarray, alpha: float = 0.05,
    ) -> tuple[float, float]:
        rng = np.random.RandomState(42)
        n = len(ra)
        diffs: list[float] = []
        for _ in range(self.BOOTSTRAP_SAMPLES):
            idx = rng.randint(0, n, size=n)
            sa = self._quick_sharpe(ra[idx])
            sb = self._quick_sharpe(rb[idx])
            diffs.append(sa - sb)
        arr = np.array(diffs)
        lower = float(np.percentile(arr, 100 * alpha / 2))
        upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return lower, upper

    @staticmethod
    def _normal_two_tail_p(z: float) -> float:
        """Two-tailed p-value from standard normal, via manual approximation."""
        # Abramowitz & Stegun approximation of the cumulative normal
        x = abs(z)
        if x > 8:
            return 0.0
        # Constants for the rational approximation
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1.0 / (1.0 + b0 * x)
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        p = 2.0 * (1.0 - cdf)
        return max(0.0, min(1.0, p))

    # ------------------------------------------------------------------ #
    #  Correlation matrix                                                #
    # ------------------------------------------------------------------ #

    def _compute_correlation_matrix(self) -> pd.DataFrame:
        if len(self._ids) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(
            {eid: self.experiment_returns[eid] for eid in self._ids}
        )
        return df.corr()

    # ------------------------------------------------------------------ #
    #  Best experiment selection                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_best(metrics: list[ExperimentMetrics]) -> str:
        if not metrics:
            return ""
        # Rank by Sharpe, break ties with Sortino
        best = max(metrics, key=lambda m: (m.sharpe, m.sortino))
        return best.experiment_id

    # ------------------------------------------------------------------ #
    #  HTML report helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _html_header() -> str:
        return (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            "<meta charset='utf-8'>\n"
            "<title>Experiment Comparison Report</title>\n"
            "<style>\n"
            "body { font-family: sans-serif; margin: 20px; }\n"
            "table { border-collapse: collapse; margin: 10px 0; }\n"
            "th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }\n"
            "th { background: #f5f5f5; }\n"
            "h2 { margin-top: 30px; }\n"
            ".best { font-weight: bold; color: green; }\n"
            "</style>\n</head>\n<body>\n"
            "<h1>Experiment Comparison Report</h1>\n"
        )

    @staticmethod
    def _html_footer() -> str:
        return "</body>\n</html>"

    def _metrics_table_html(self, result: CompareResult) -> str:
        cols = [
            "Experiment", "Sharpe", "Sortino", "Calmar", "Total Return",
            "Annual Return", "Max DD", "Win Rate", "Avg Trade Dur", "Profit Factor",
        ]
        rows: list[str] = []
        for m in result.experiment_metrics:
            cls = ' class="best"' if m.experiment_id == result.best_experiment else ""
            rows.append(
                f"<tr{cls}>"
                f"<td style='text-align:left'>{m.experiment_id}</td>"
                f"<td>{m.sharpe:.3f}</td>"
                f"<td>{m.sortino:.3f}</td>"
                f"<td>{m.calmar:.3f}</td>"
                f"<td>{m.total_return:.4f}</td>"
                f"<td>{m.annual_return:.4f}</td>"
                f"<td>{m.max_dd:.4f}</td>"
                f"<td>{m.win_rate:.3f}</td>"
                f"<td>{m.avg_trade_duration:.1f}</td>"
                f"<td>{m.profit_factor:.3f}</td>"
                "</tr>"
            )
        header = "".join(f"<th>{c}</th>" for c in cols)
        return (
            "<h2>Experiment Metrics</h2>\n"
            f"<table><thead><tr>{header}</tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n"
        )

    def _equity_curve_svg(self, result: CompareResult) -> str:
        return self._line_chart_svg(
            title="Equity Curves",
            series={
                m.experiment_id: (1 + self.experiment_returns[m.experiment_id].dropna()).cumprod()
                for m in result.experiment_metrics
                if len(self.experiment_returns[m.experiment_id].dropna()) > 0
            },
            ylabel="Cumulative Return",
        )

    def _drawdown_svg(self, result: CompareResult) -> str:
        dd_series: dict[str, pd.Series] = {}
        for m in result.experiment_metrics:
            r = self.experiment_returns[m.experiment_id].dropna()
            if len(r) == 0:
                continue
            cum = (1 + r).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            dd_series[m.experiment_id] = dd
        return self._line_chart_svg(
            title="Drawdowns", series=dd_series, ylabel="Drawdown",
        )

    def _line_chart_svg(
        self, title: str, series: dict[str, pd.Series], ylabel: str,
    ) -> str:
        if not series:
            return f"<h2>{title}</h2><p>No data</p>"
        w, h = 700, 300
        pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 30
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b

        all_vals = np.concatenate([s.values for s in series.values()])
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        max_len = max(len(s) for s in series.values())

        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        paths: list[str] = []
        legend_items: list[str] = []
        for ci, (name, s) in enumerate(series.items()):
            col = colours[ci % len(colours)]
            pts: list[str] = []
            for i, v in enumerate(s.values):
                x = pad_l + (i / max(max_len - 1, 1)) * pw
                y = pad_t + ph - ((v - ymin) / (ymax - ymin)) * ph
                pts.append(f"{x:.1f},{y:.1f}")
            paths.append(
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{col}" stroke-width="1.5"/>'
            )
            lx = pad_l + 10 + ci * 120
            legend_items.append(
                f'<rect x="{lx}" y="{h - 12}" width="10" height="10" fill="{col}"/>'
                f'<text x="{lx + 14}" y="{h - 3}" font-size="10">{name}</text>'
            )

        # Y-axis labels
        ylabels: list[str] = []
        for i in range(5):
            yv = ymin + (ymax - ymin) * i / 4
            yp = pad_t + ph - (i / 4) * ph
            ylabels.append(
                f'<text x="{pad_l - 5}" y="{yp + 4}" font-size="9" '
                f'text-anchor="end">{yv:.3f}</text>'
            )

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n'
            f'<text x="{w // 2}" y="18" text-anchor="middle" font-size="13" '
            f'font-weight="bold">{title}</text>\n'
            f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" '
            f'stroke="#ccc"/>\n'
            f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" '
            f'y2="{pad_t + ph}" stroke="#ccc"/>\n'
            + "\n".join(ylabels) + "\n"
            + "\n".join(paths) + "\n"
            + "\n".join(legend_items) + "\n"
            + "</svg>\n"
        )
        return f"<h2>{title}</h2>\n{svg}\n"

    def _pair_comparisons_html(self, result: CompareResult) -> str:
        if not result.pair_comparisons:
            return "<h2>Pair Comparisons</h2><p>N/A</p>"
        cols = ["Exp A", "Exp B", "Mean Diff", "t-stat", "p-value",
                "Sharpe Diff", "CI Lower", "CI Upper"]
        header = "".join(f"<th>{c}</th>" for c in cols)
        rows: list[str] = []
        for p in result.pair_comparisons:
            rows.append(
                f"<tr><td>{p.exp_a}</td><td>{p.exp_b}</td>"
                f"<td>{p.mean_diff:.6f}</td><td>{p.t_stat:.3f}</td>"
                f"<td>{p.p_value:.4f}</td><td>{p.sharpe_diff:.3f}</td>"
                f"<td>{p.ci_lower:.3f}</td><td>{p.ci_upper:.3f}</td></tr>"
            )
        return (
            "<h2>Pair Comparisons</h2>\n"
            f"<table><thead><tr>{header}</tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n"
        )

    def _correlation_heatmap_svg(self, result: CompareResult) -> str:
        corr = result.correlation_matrix
        if corr.empty:
            return "<h2>Correlation Heatmap</h2><p>No data</p>"
        n = len(corr)
        cell = 60
        pad_l, pad_t = 80, 80
        w = pad_l + n * cell + 20
        h = pad_t + n * cell + 20

        cells: list[str] = []
        for i, row_name in enumerate(corr.index):
            for j, col_name in enumerate(corr.columns):
                v = corr.iloc[i, j]
                # Colour: blue (1) → white (0) → red (-1)
                if np.isnan(v):
                    fill = "#ccc"
                elif v >= 0:
                    g = int(255 * (1 - v))
                    fill = f"rgb({g},{g},255)"
                else:
                    g = int(255 * (1 + v))
                    fill = f"rgb(255,{g},{g})"
                x = pad_l + j * cell
                y = pad_t + i * cell
                cells.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="white"/>'
                    f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" '
                    f'text-anchor="middle" font-size="10">{v:.2f}</text>'
                )

        # Labels
        labels: list[str] = []
        for i, name in enumerate(corr.columns):
            x = pad_l + i * cell + cell // 2
            labels.append(
                f'<text x="{x}" y="{pad_t - 8}" text-anchor="middle" '
                f'font-size="10" transform="rotate(-45,{x},{pad_t - 8})">{name}</text>'
            )
        for i, name in enumerate(corr.index):
            y = pad_t + i * cell + cell // 2 + 4
            labels.append(
                f'<text x="{pad_l - 5}" y="{y}" text-anchor="end" font-size="10">'
                f'{name}</text>'
            )

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n'
            + "\n".join(cells) + "\n"
            + "\n".join(labels) + "\n"
            + "</svg>\n"
        )
        return f"<h2>Correlation Heatmap</h2>\n{svg}\n"

    def _monthly_returns_html(self, result: CompareResult) -> str:
        parts: list[str] = ["<h2>Monthly Returns</h2>"]
        for m in result.experiment_metrics:
            r = self.experiment_returns[m.experiment_id].dropna()
            if len(r) == 0 or not isinstance(r.index, pd.DatetimeIndex):
                parts.append(f"<h3>{m.experiment_id}</h3><p>No date index</p>")
                continue
            monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            if len(monthly) == 0:
                parts.append(f"<h3>{m.experiment_id}</h3><p>No data</p>")
                continue
            pivot = pd.DataFrame({
                "Year": monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            })
            table = pivot.pivot_table(
                values="Return", index="Year", columns="Month", aggfunc="sum",
            )
            table.columns = [
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
                for c in table.columns
            ]
            parts.append(f"<h3>{m.experiment_id}</h3>")
            parts.append(table.to_html(float_format="%.4f", na_rep=""))
        return "\n".join(parts)
