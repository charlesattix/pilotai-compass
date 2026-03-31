"""
compass/experiment_pipeline.py — Automated batch experiment runner.

Provides a structured pipeline for running multiple strategy experiments
in parallel, collecting metrics, performing walk-forward validation, and
generating comparison reports.

Typical usage::

    from compass.experiment_pipeline import ExperimentConfig, ExperimentPipeline

    configs = [
        ExperimentConfig(
            experiment_id="EXP-500",
            name="Iron Condor Narrow",
            strategy_class="IronCondorStrategy",
            ticker="SPY",
            config_overrides={"spread_width": 5, "dte_target": 45},
            description="Narrow IC on SPY with 45 DTE",
        ),
        ExperimentConfig(
            experiment_id="EXP-501",
            name="Put Credit Spread Wide",
            strategy_class="CreditSpreadStrategy",
            ticker="QQQ",
            config_overrides={"spread_width": 10},
        ),
    ]

    pipeline = ExperimentPipeline(configs, data_dir="data/", output_dir="experiments/pipeline_results/")
    pipeline.run_all()
    comparison = pipeline.compare()
    pipeline.to_html("experiments/pipeline_results/comparison.html")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes:
        experiment_id: Unique identifier (e.g. "EXP-500").
        name: Human-readable experiment name.
        strategy_class: Strategy class name (resolved via strategy_factory).
        ticker: Underlying ticker symbol.
        config_overrides: Strategy parameter overrides applied on top of defaults.
        description: Free-text description of the experiment hypothesis.
        start_date: Backtest start date (ISO format string). Defaults to "2020-01-01".
        end_date: Backtest end date (ISO format string). Defaults to "2025-12-31".
        starting_capital: Initial portfolio value.
    """

    experiment_id: str
    name: str
    strategy_class: str
    ticker: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    starting_capital: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.strategy_class:
            raise ValueError("strategy_class must not be empty")
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.starting_capital <= 0:
            raise ValueError(
                f"starting_capital must be positive, got {self.starting_capital}"
            )


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        config: The experiment configuration that produced this result.
        backtest_results: Raw results dict from PortfolioBacktester.run().
        walk_forward: Walk-forward validation results, or None if skipped.
        metrics: Extracted key metrics for easy comparison.
        timestamp: ISO timestamp when the experiment completed.
        status: "completed", "failed", or "skipped".
        error: Error message if status is "failed".
    """

    config: ExperimentConfig
    backtest_results: Optional[Dict[str, Any]] = None
    walk_forward: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d = {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "status": self.status,
            "error": self.error,
        }
        if self.walk_forward is not None:
            d["walk_forward"] = self.walk_forward
        return d


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(backtest_results: Dict[str, Any]) -> Dict[str, float]:
    """Pull key comparison metrics from a PortfolioBacktester result dict.

    Returns:
        Dict with keys: sharpe, cagr_pct, max_dd_pct, win_rate, total_trades,
        profit_factor, total_pnl, avg_win, avg_loss, return_pct.
    """
    combined = backtest_results.get("combined", {})

    total_trades = combined.get("total_trades", 0)
    winning = combined.get("winning_trades", 0)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0

    # Compute CAGR from yearly data if available
    yearly = backtest_results.get("yearly", {})
    cagr = _compute_cagr_from_yearly(yearly, combined)

    return {
        "sharpe": float(combined.get("sharpe_ratio", 0.0)),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(float(combined.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": int(total_trades),
        "profit_factor": float(combined.get("profit_factor", 0.0)),
        "total_pnl": float(combined.get("total_pnl", 0.0)),
        "avg_win": float(combined.get("avg_win", 0.0)),
        "avg_loss": float(combined.get("avg_loss", 0.0)),
        "return_pct": float(combined.get("return_pct", 0.0)),
    }


def _compute_cagr_from_yearly(
    yearly: Dict[str, Any],
    combined: Dict[str, Any],
) -> float:
    """Compute CAGR from yearly returns or fall back to combined return."""
    if not yearly:
        ret = combined.get("return_pct", 0.0)
        return float(ret)

    n_years = len(yearly)
    if n_years == 0:
        return 0.0

    # Chain yearly returns
    cumulative = 1.0
    for year_data in yearly.values():
        yr_ret = year_data.get("return_pct", 0.0)
        if isinstance(yr_ret, (int, float)):
            cumulative *= 1 + yr_ret / 100.0

    if cumulative <= 0 or n_years == 0:
        return 0.0

    cagr = (cumulative ** (1.0 / n_years) - 1) * 100
    return float(cagr)


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


def run_walk_forward(
    backtest_results: Dict[str, Any],
    pass_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """Run walk-forward validation on yearly backtest results.

    Uses the validation module's WalkForwardValidator if yearly data is
    available.  Returns None if insufficient data.
    """
    yearly = backtest_results.get("yearly", {})
    if len(yearly) < 3:
        logger.info("Skipping walk-forward: need >= 3 years of data, got %d", len(yearly))
        return None

    try:
        from validation import WalkForwardValidator
    except ImportError:
        logger.warning("validation module not available; skipping walk-forward")
        return None

    # Build results_by_year dict expected by WalkForwardValidator
    results_by_year: Dict[str, Dict] = {}
    for year_str, year_data in sorted(yearly.items()):
        results_by_year[year_str] = {
            "return_pct": year_data.get("return_pct", 0.0),
            "max_drawdown": year_data.get("max_drawdown", 0.0),
            "sharpe_ratio": year_data.get("sharpe_ratio", 0.0),
            "total_trades": year_data.get("total_trades", 0),
        }

    validator = WalkForwardValidator(pass_threshold=pass_threshold)
    result = validator.run(results_by_year)

    return dict(result) if result else None


# ---------------------------------------------------------------------------
# ExperimentPipeline
# ---------------------------------------------------------------------------


class ExperimentPipeline:
    """Automated batch experiment runner.

    Runs multiple strategy backtests, collects metrics, performs walk-forward
    validation, and generates comparison reports.

    Args:
        configs: List of ExperimentConfig instances to run.
        data_dir: Path to market data directory.
        output_dir: Path where results and reports are saved.
        strategy_factory: Optional callable ``(config) -> (strategy_name, strategy_instance)``
            for resolving strategy_class strings. If None, uses a default
            resolver that imports from the strategies package.
    """

    def __init__(
        self,
        configs: List[ExperimentConfig],
        data_dir: str = "data/",
        output_dir: str = "experiments/pipeline_results/",
        strategy_factory: Optional[Callable[[ExperimentConfig], Tuple[str, Any]]] = None,
    ) -> None:
        if not configs:
            raise ValueError("configs must not be empty")

        # Check for duplicate experiment IDs
        ids = [c.experiment_id for c in configs]
        dupes = [eid for eid in ids if ids.count(eid) > 1]
        if dupes:
            raise ValueError(f"Duplicate experiment_id(s): {set(dupes)}")

        self.configs = list(configs)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.strategy_factory = strategy_factory
        self.results: List[ExperimentResult] = []
        self._results_by_id: Dict[str, ExperimentResult] = {}

    @property
    def completed(self) -> List[ExperimentResult]:
        """Return only successfully completed results."""
        return [r for r in self.results if r.status == "completed"]

    @property
    def failed(self) -> List[ExperimentResult]:
        """Return only failed results."""
        return [r for r in self.results if r.status == "failed"]

    def run_all(self) -> List[ExperimentResult]:
        """Run all configured experiments sequentially.

        Returns:
            List of ExperimentResult instances (one per config).
        """
        logger.info("ExperimentPipeline: running %d experiments", len(self.configs))
        self.results = []
        self._results_by_id = {}

        for config in self.configs:
            result = self.run_single(config.experiment_id)
            self.results.append(result)
            self._results_by_id[config.experiment_id] = result

        n_ok = len(self.completed)
        n_fail = len(self.failed)
        logger.info(
            "ExperimentPipeline: %d completed, %d failed out of %d",
            n_ok, n_fail, len(self.configs),
        )
        return self.results

    def run_single(self, experiment_id: str) -> ExperimentResult:
        """Run a single experiment by its ID.

        Args:
            experiment_id: The experiment_id to run.

        Returns:
            ExperimentResult with status "completed" or "failed".

        Raises:
            KeyError: If experiment_id is not found in configs.
        """
        config = self._get_config(experiment_id)
        logger.info("Running experiment %s: %s", config.experiment_id, config.name)

        try:
            backtest_results = self._run_backtest(config)
            metrics = extract_metrics(backtest_results)
            wf_result = run_walk_forward(backtest_results)

            result = ExperimentResult(
                config=config,
                backtest_results=backtest_results,
                walk_forward=wf_result,
                metrics=metrics,
                timestamp=datetime.utcnow().isoformat(),
                status="completed",
            )
        except Exception as exc:
            logger.error(
                "Experiment %s failed: %s", config.experiment_id, exc, exc_info=True,
            )
            result = ExperimentResult(
                config=config,
                timestamp=datetime.utcnow().isoformat(),
                status="failed",
                error=str(exc),
            )

        self._results_by_id[experiment_id] = result
        return result

    def compare(self) -> pd.DataFrame:
        """Generate a side-by-side comparison DataFrame of all completed experiments.

        Columns: experiment_id, name, ticker, sharpe, cagr_pct, max_dd_pct,
        win_rate, total_trades, profit_factor, total_pnl, wf_consistent.

        Returns:
            pd.DataFrame sorted by Sharpe ratio descending.
        """
        rows: List[Dict[str, Any]] = []
        for result in self.completed:
            row = {
                "experiment_id": result.config.experiment_id,
                "name": result.config.name,
                "ticker": result.config.ticker,
                "strategy_class": result.config.strategy_class,
            }
            row.update(result.metrics)

            # Walk-forward consistency flag
            if result.walk_forward is not None:
                row["wf_consistent"] = result.walk_forward.get("consistent", False)
                row["wf_pass_rate"] = result.walk_forward.get("pass_rate", 0.0)
            else:
                row["wf_consistent"] = None
                row["wf_pass_rate"] = None

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        return df

    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Retrieve result for a specific experiment, or None if not yet run."""
        return self._results_by_id.get(experiment_id)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise all results to JSON.

        Args:
            path: Optional file path. If provided, writes to file and returns
                the path. If None, returns the JSON string.

        Returns:
            JSON string or file path.
        """
        payload = {
            "pipeline_run": {
                "timestamp": datetime.utcnow().isoformat(),
                "n_experiments": len(self.configs),
                "n_completed": len(self.completed),
                "n_failed": len(self.failed),
            },
            "experiments": [r.to_dict() for r in self.results],
        }

        # Add comparison table if we have results
        if self.completed:
            comparison_df = self.compare()
            payload["comparison"] = comparison_df.to_dict(orient="records")

        json_str = json.dumps(payload, indent=2, default=_json_safe)

        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json_str)
            logger.info("Saved JSON results → %s", out)
            return str(out)

        return json_str

    def to_html(self, path: Optional[str] = None) -> str:
        """Generate an HTML comparison report.

        Args:
            path: Optional file path. If provided, writes to file and returns
                the path. If None, returns the HTML string.

        Returns:
            HTML string or file path.
        """
        html = _render_html_report(self)

        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
            logger.info("Saved HTML report → %s", out)
            return str(out)

        return html

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_config(self, experiment_id: str) -> ExperimentConfig:
        """Look up config by experiment_id."""
        for c in self.configs:
            if c.experiment_id == experiment_id:
                return c
        raise KeyError(f"No config with experiment_id={experiment_id!r}")

    def _run_backtest(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Execute a single backtest for *config*.

        Resolves the strategy class, instantiates PortfolioBacktester, and
        runs the simulation.
        """
        strategy_name, strategy_instance = self._resolve_strategy(config)

        from engine.portfolio_backtester import PortfolioBacktester

        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")

        backtester = PortfolioBacktester(
            strategies=[(strategy_name, strategy_instance)],
            tickers=[config.ticker],
            start_date=start_dt,
            end_date=end_dt,
            starting_capital=config.starting_capital,
        )

        results = backtester.run()
        return results

    def _resolve_strategy(self, config: ExperimentConfig) -> Tuple[str, Any]:
        """Resolve strategy_class string to a (name, instance) tuple."""
        if self.strategy_factory is not None:
            return self.strategy_factory(config)

        # Default: try importing from strategies package
        try:
            from shared.strategy_factory import create_strategy
            instance = create_strategy(config.strategy_class, **config.config_overrides)
            return (config.name, instance)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot resolve strategy_class={config.strategy_class!r}. "
                f"Provide a strategy_factory or install the strategies package. "
                f"Error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """JSON default handler for numpy and other non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# HTML report renderer
# ---------------------------------------------------------------------------


def _render_html_report(pipeline: ExperimentPipeline) -> str:
    """Generate a standalone HTML comparison report."""
    comparison_df = pipeline.compare()

    # Summary stats
    n_total = len(pipeline.configs)
    n_ok = len(pipeline.completed)
    n_fail = len(pipeline.failed)

    # Build comparison table
    if not comparison_df.empty:
        comparison_table = _df_to_html_table(comparison_df)
    else:
        comparison_table = "<p><em>No completed experiments to compare.</em></p>"

    # Per-experiment detail cards
    detail_cards = ""
    for result in pipeline.results:
        detail_cards += _experiment_card(result)

    # Failed experiments
    failed_section = ""
    if pipeline.failed:
        failed_rows = "".join(
            f"<tr><td>{r.config.experiment_id}</td>"
            f"<td>{r.config.name}</td>"
            f"<td style='color:#dc3545'>{r.error or 'Unknown error'}</td></tr>"
            for r in pipeline.failed
        )
        failed_section = (
            f"<h2>Failed Experiments</h2>"
            f"<table class='data-table'><thead><tr>"
            f"<th>ID</th><th>Name</th><th>Error</th>"
            f"</tr></thead><tbody>{failed_rows}</tbody></table>"
        )

    banner_bg = "#d4edda" if n_fail == 0 else "#f8d7da"
    banner_fg = "#155724" if n_fail == 0 else "#721c24"
    banner_msg = (
        f"{n_ok}/{n_total} experiments completed successfully"
        if n_fail == 0
        else f"{n_fail}/{n_total} experiments failed"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Pipeline Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #ffffff; color: #212529; line-height: 1.5;
    padding: 32px 24px; max-width: 1200px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.7em; font-weight: 700; margin-bottom: 6px; }}
  h2 {{ font-size: 1.2em; font-weight: 600; margin: 32px 0 14px;
        padding-bottom: 8px; border-bottom: 2px solid #dee2e6; }}
  .subtitle {{ color: #666; font-size: 0.95em; margin-bottom: 24px; }}
  .banner {{
    border-radius: 6px; padding: 12px 18px; margin-bottom: 28px;
    font-size: 0.95em; font-weight: 500;
  }}
  .data-table {{
    border-collapse: collapse; width: 100%; font-size: 0.88em; margin-bottom: 24px;
  }}
  .data-table th {{
    background: #f8f9fa; padding: 10px 12px; text-align: left;
    border-bottom: 2px solid #dee2e6; font-weight: 600;
  }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #dee2e6; }}
  .data-table tr:hover {{ background: #f8f9fa; }}
  .card {{
    border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;
    background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 16px;
  }}
  .card h3 {{ font-size: 1em; margin-bottom: 8px; }}
  .metric {{ display: inline-block; margin-right: 24px; margin-bottom: 8px; }}
  .metric-label {{ font-size: 0.8em; color: #666; }}
  .metric-value {{ font-size: 1.1em; font-weight: 600; }}
  .status-ok {{ color: #155724; }}
  .status-fail {{ color: #dc3545; }}
</style>
</head>
<body>

<h1>Experiment Pipeline Report</h1>
<p class="subtitle">
  {n_total} experiments &nbsp;|&nbsp;
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
</p>

<div class="banner" style="background:{banner_bg};color:{banner_fg}">
  {banner_msg}
</div>

<h2>Side-by-Side Comparison</h2>
{comparison_table}

{failed_section}

<h2>Experiment Details</h2>
{detail_cards}

<hr style="margin:36px 0;border:none;border-top:1px solid #dee2e6">
<p style="font-size:0.78em;color:#999">
  Generated by <code>compass/experiment_pipeline.py</code>
</p>
</body>
</html>"""


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to an HTML table with styling."""
    display_cols = [
        ("experiment_id", "ID"),
        ("name", "Name"),
        ("ticker", "Ticker"),
        ("sharpe", "Sharpe"),
        ("cagr_pct", "CAGR %"),
        ("max_dd_pct", "Max DD %"),
        ("win_rate", "Win Rate %"),
        ("total_trades", "Trades"),
        ("profit_factor", "Profit Factor"),
        ("wf_consistent", "WF Pass"),
    ]

    # Filter to columns that exist
    cols = [(c, label) for c, label in display_cols if c in df.columns]

    header = "".join(f"<th>{label}</th>" for _, label in cols)
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col, _ in cols:
            val = row[col]
            if col == "sharpe":
                color = "#155724" if val >= 1.0 else ("#856404" if val >= 0.5 else "#dc3545")
                cells += f"<td style='font-weight:600;color:{color}'>{val:.3f}</td>"
            elif col == "max_dd_pct":
                cells += f"<td>{val:.1f}%</td>"
            elif col == "cagr_pct":
                cells += f"<td>{val:.1f}%</td>"
            elif col == "win_rate":
                cells += f"<td>{val:.1f}%</td>"
            elif col == "profit_factor":
                cells += f"<td>{val:.2f}</td>"
            elif col == "wf_consistent":
                if val is None:
                    cells += "<td style='color:#999'>N/A</td>"
                else:
                    icon = "PASS" if val else "FAIL"
                    color = "#155724" if val else "#dc3545"
                    cells += f"<td style='color:{color};font-weight:600'>{icon}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    return (
        f"<table class='data-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _experiment_card(result: ExperimentResult) -> str:
    """Render a single experiment detail card."""
    cfg = result.config
    status_class = "status-ok" if result.status == "completed" else "status-fail"
    status_label = result.status.upper()

    if result.status == "failed":
        return (
            f"<div class='card'>"
            f"<h3>{cfg.experiment_id}: {cfg.name} "
            f"<span class='{status_class}'>({status_label})</span></h3>"
            f"<p style='color:#dc3545'>{result.error}</p>"
            f"</div>"
        )

    m = result.metrics
    metrics_html = "".join(
        f"<div class='metric'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{fmt}</div></div>"
        for label, fmt in [
            ("Sharpe", f"{m.get('sharpe', 0):.3f}"),
            ("CAGR", f"{m.get('cagr_pct', 0):.1f}%"),
            ("Max DD", f"{m.get('max_dd_pct', 0):.1f}%"),
            ("Win Rate", f"{m.get('win_rate', 0):.1f}%"),
            ("Trades", f"{m.get('total_trades', 0)}"),
            ("Profit Factor", f"{m.get('profit_factor', 0):.2f}"),
            ("Total PnL", f"${m.get('total_pnl', 0):,.0f}"),
        ]
    )

    wf_html = ""
    if result.walk_forward:
        wf_ok = result.walk_forward.get("consistent", False)
        wf_label = "PASS" if wf_ok else "FAIL"
        wf_color = "#155724" if wf_ok else "#dc3545"
        wf_rate = result.walk_forward.get("pass_rate", 0)
        wf_html = (
            f"<div class='metric'>"
            f"<div class='metric-label'>Walk-Forward</div>"
            f"<div class='metric-value' style='color:{wf_color}'>"
            f"{wf_label} ({wf_rate:.0%})</div></div>"
        )

    desc = f"<p style='color:#666;font-size:0.85em;margin-bottom:12px'>{cfg.description}</p>" if cfg.description else ""

    return (
        f"<div class='card'>"
        f"<h3>{cfg.experiment_id}: {cfg.name} "
        f"<span class='{status_class}'>({status_label})</span> "
        f"&mdash; {cfg.ticker} / {cfg.strategy_class}</h3>"
        f"{desc}"
        f"{metrics_html}{wf_html}"
        f"</div>"
    )
