from __future__ import annotations

import math
import html as html_mod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100_000.0,
    "transaction_cost_bps": 5.0,
    "spread_cost_bps": 2.0,
    "max_position_pct": 0.20,
    "rebalance_freq": "daily",
    "stop_loss_pct": 0.05,
    "n_splits": 5,
    "mc_simulations": 1000,
    "mc_confidence_levels": [0.05, 0.50, 0.95],
    "train_ratio": 0.7,
}

REQUIRED_CONFIG_KEYS = [
    "start_date",
    "end_date",
    "initial_capital",
    "transaction_cost_bps",
    "spread_cost_bps",
    "max_position_pct",
    "rebalance_freq",
]


# ---------------------------------------------------------------------------
# Data‑classes for results
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    exit_price: Optional[float]
    pnl: float = 0.0
    costs: float = 0.0
    exit_reason: str = ""


@dataclass
class WalkForwardResult:
    split_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float


@dataclass
class MonteCarloResult:
    sharpe_ci: Dict[float, float] = field(default_factory=dict)
    returns_ci: Dict[float, float] = field(default_factory=dict)
    max_dd_ci: Dict[float, float] = field(default_factory=dict)
    simulated_sharpes: Optional[np.ndarray] = None
    simulated_returns: Optional[np.ndarray] = None
    simulated_max_dds: Optional[np.ndarray] = None


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: List[Trade] = field(default_factory=list)
    daily_pnl: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    metrics: Dict[str, Any] = field(default_factory=dict)
    walk_forward_results: List[WalkForwardResult] = field(default_factory=list)
    monte_carlo: MonteCarloResult = field(default_factory=MonteCarloResult)
    regime_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge with defaults and validate required keys / value constraints."""
    merged = {**DEFAULT_CONFIG, **config}
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in merged]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    if merged["initial_capital"] <= 0:
        raise ValueError("initial_capital must be positive")
    if not (0 < merged["max_position_pct"] <= 1.0):
        raise ValueError("max_position_pct must be in (0, 1]")
    if merged["transaction_cost_bps"] < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if merged["spread_cost_bps"] < 0:
        raise ValueError("spread_cost_bps must be >= 0")

    allowed_freq = {"daily", "weekly", "monthly"}
    if merged["rebalance_freq"] not in allowed_freq:
        raise ValueError(f"rebalance_freq must be one of {allowed_freq}")

    # Date parsing
    merged["start_date"] = pd.Timestamp(merged["start_date"])
    merged["end_date"] = pd.Timestamp(merged["end_date"])
    if merged["start_date"] >= merged["end_date"]:
        raise ValueError("start_date must be before end_date")

    return merged


def compute_max_drawdown(equity: pd.Series) -> float:
    """Return maximum drawdown as a positive fraction."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(-dd.min()) if not dd.empty else 0.0


def compute_sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.empty:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def compute_sortino(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * math.sqrt(periods_per_year))


def compute_calmar(total_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return total_return / max_dd


def compute_trade_impact(trade_value: float, avg_daily_volume: float = 1e8) -> float:
    """Square‑root market impact model: impact proportional to sqrt(trade_value / ADV)."""
    if avg_daily_volume <= 0:
        return 0.0
    return 0.1 * math.sqrt(abs(trade_value) / avg_daily_volume)


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set:
    """Return the set of dates on which rebalancing is allowed."""
    if freq == "daily":
        return set(dates)
    elif freq == "weekly":
        # last trading day of each week
        s = pd.Series(dates, index=dates)
        return set(s.resample("W").last().dropna().values)
    elif freq == "monthly":
        s = pd.Series(dates, index=dates)
        return set(s.resample("ME").last().dropna().values)
    return set(dates)


# ---------------------------------------------------------------------------
# Walk‑forward splitter
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    n_splits: int,
    train_ratio: float = 0.7,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate expanding or rolling walk‑forward splits."""
    if n_splits <= 0:
        raise ValueError("n_splits must be > 0")
    n = len(dates)
    if n < n_splits * 2:
        raise ValueError("Not enough data for requested walk-forward splits")

    split_size = n // n_splits
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(n_splits):
        end_idx = (i + 1) * split_size if i < n_splits - 1 else n
        start_idx = i * split_size
        seg = dates[start_idx:end_idx]
        train_len = max(1, int(len(seg) * train_ratio))
        train = seg[:train_len]
        test = seg[train_len:]
        if len(test) == 0:
            test = seg[-1:]
        splits.append((train, test))
    return splits


# ---------------------------------------------------------------------------
# SystematicBacktester
# ---------------------------------------------------------------------------

class SystematicBacktester:
    """End‑to‑end systematic backtesting framework."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = validate_config(config or {})
        self._feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
        self._signal_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None
        self._data: Optional[pd.DataFrame] = None
        self._regime_series: Optional[pd.Series] = None

    # -- pipeline registration ------------------------------------------------

    def load_data(self, df: pd.DataFrame) -> SystematicBacktester:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        mask = (df.index >= self.config["start_date"]) & (
            df.index <= self.config["end_date"]
        )
        self._data = df.loc[mask].copy().sort_index()
        if self._data.empty:
            raise ValueError("No data in the configured date range")
        return self

    def set_feature_fn(
        self, fn: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> SystematicBacktester:
        self._feature_fn = fn
        return self

    def set_signal_fn(
        self, fn: Callable[[pd.DataFrame], pd.Series]
    ) -> SystematicBacktester:
        self._signal_fn = fn
        return self

    def set_regime_series(self, series: pd.Series) -> SystematicBacktester:
        self._regime_series = series
        return self

    # -- cost model -----------------------------------------------------------

    def _compute_trade_cost(self, trade_value: float) -> float:
        """Total single‑leg cost: spread + impact + commission."""
        spread_cost = abs(trade_value) * self.config["spread_cost_bps"] / 10_000
        commission = abs(trade_value) * self.config["transaction_cost_bps"] / 10_000
        impact = abs(trade_value) * compute_trade_impact(trade_value)
        return spread_cost + commission + impact

    # -- simulation engine ----------------------------------------------------

    def _run_simulation(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> Tuple[pd.Series, List[Trade], pd.DataFrame]:
        """Core simulation loop returning equity curve, trades, and daily P&L df."""
        cfg = self.config
        capital = cfg["initial_capital"]
        max_pos_pct = cfg["max_position_pct"]
        stop_loss_pct = cfg.get("stop_loss_pct", 0.05)

        rebal_dates = _rebalance_dates(data.index, cfg["rebalance_freq"])

        equity: Dict[pd.Timestamp, float] = {}
        pnl_records: List[Dict[str, Any]] = []
        trades: List[Trade] = []

        position = 0.0
        entry_price = 0.0
        entry_date: Optional[pd.Timestamp] = None
        direction = ""

        prices = data["close"]

        for dt in data.index:
            price = float(prices.loc[dt])
            daily_alpha = 0.0
            daily_cost = 0.0
            daily_slippage = 0.0

            # Mark‑to‑market
            if position != 0:
                prev_price = entry_price if len(equity) == 0 else float(
                    prices.iloc[max(0, data.index.get_loc(dt) - 1)]
                )
                if direction == "long":
                    daily_alpha = position * (price - prev_price)
                else:
                    daily_alpha = position * (prev_price - price)

            # Stop‑loss check
            if position != 0 and entry_price != 0:
                if direction == "long":
                    unrealised = (price - entry_price) / entry_price
                else:
                    unrealised = (entry_price - price) / entry_price
                if unrealised < -stop_loss_pct:
                    # Close position
                    trade_val = position * price
                    cost = self._compute_trade_cost(trade_val)
                    daily_cost += cost
                    if direction == "long":
                        trade_pnl = position * (price - entry_price) - cost
                    else:
                        trade_pnl = position * (entry_price - price) - cost
                    trades.append(
                        Trade(
                            entry_date=entry_date,  # type: ignore[arg-type]
                            exit_date=dt,
                            direction=direction,
                            size=position,
                            entry_price=entry_price,
                            exit_price=price,
                            pnl=trade_pnl,
                            costs=cost,
                            exit_reason="stop_loss",
                        )
                    )
                    capital += trade_pnl
                    position = 0.0
                    entry_price = 0.0
                    direction = ""

            # Signal & rebalance
            sig = float(signals.get(dt, 0.0))
            if dt in rebal_dates and position == 0 and sig != 0:
                target_notional = capital * max_pos_pct
                target_shares = target_notional / price
                trade_val = target_shares * price
                cost = self._compute_trade_cost(trade_val)
                daily_cost += cost
                daily_slippage = abs(trade_val) * compute_trade_impact(trade_val)
                position = target_shares
                entry_price = price
                entry_date = dt
                direction = "long" if sig > 0 else "short"
                capital -= cost  # pay costs up‑front

            elif dt in rebal_dates and position != 0 and sig == 0:
                # Close on zero signal
                trade_val = position * price
                cost = self._compute_trade_cost(trade_val)
                daily_cost += cost
                daily_slippage = abs(trade_val) * compute_trade_impact(trade_val)
                if direction == "long":
                    trade_pnl = position * (price - entry_price) - cost
                else:
                    trade_pnl = position * (entry_price - price) - cost
                trades.append(
                    Trade(
                        entry_date=entry_date,  # type: ignore[arg-type]
                        exit_date=dt,
                        direction=direction,
                        size=position,
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=trade_pnl,
                        costs=cost,
                        exit_reason="signal",
                    )
                )
                capital += trade_pnl
                position = 0.0
                entry_price = 0.0
                direction = ""

            # Record equity (capital + unrealised)
            unrealised_val = 0.0
            if position != 0:
                if direction == "long":
                    unrealised_val = position * (price - entry_price)
                else:
                    unrealised_val = position * (entry_price - price)
            equity[dt] = capital + unrealised_val

            pnl_records.append(
                {
                    "date": dt,
                    "alpha": daily_alpha,
                    "costs": daily_cost,
                    "slippage": daily_slippage,
                    "total": daily_alpha - daily_cost,
                }
            )

        eq_series = pd.Series(equity, name="equity").sort_index()
        pnl_df = pd.DataFrame(pnl_records).set_index("date")
        return eq_series, trades, pnl_df

    # -- walk‑forward --------------------------------------------------------

    def _walk_forward(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> List[WalkForwardResult]:
        n_splits = self.config.get("n_splits", 5)
        train_ratio = self.config.get("train_ratio", 0.7)
        try:
            splits = walk_forward_splits(data.index, n_splits, train_ratio)
        except ValueError:
            return []

        results: List[WalkForwardResult] = []
        for idx, (train_idx, test_idx) in enumerate(splits):
            train_sig = signals.reindex(train_idx).fillna(0)
            test_sig = signals.reindex(test_idx).fillna(0)

            train_data = data.loc[train_idx]
            test_data = data.loc[test_idx]

            if train_data.empty or test_data.empty:
                continue

            eq_train, _, _ = self._run_simulation(train_data, train_sig)
            eq_test, _, _ = self._run_simulation(test_data, test_sig)

            ret_train = eq_train.pct_change().dropna()
            ret_test = eq_test.pct_change().dropna()

            is_sharpe = compute_sharpe(ret_train)
            oos_sharpe = compute_sharpe(ret_test)

            is_return = (
                float((eq_train.iloc[-1] / eq_train.iloc[0]) - 1)
                if len(eq_train) > 1
                else 0.0
            )
            oos_return = (
                float((eq_test.iloc[-1] / eq_test.iloc[0]) - 1)
                if len(eq_test) > 1
                else 0.0
            )

            results.append(
                WalkForwardResult(
                    split_index=idx,
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                    in_sample_sharpe=is_sharpe,
                    out_of_sample_sharpe=oos_sharpe,
                    in_sample_return=is_return,
                    out_of_sample_return=oos_return,
                )
            )
        return results

    # -- Monte Carlo ----------------------------------------------------------

    def _monte_carlo(
        self, daily_returns: pd.Series, seed: int = 42
    ) -> MonteCarloResult:
        n_sim = self.config.get("mc_simulations", 1000)
        ci_levels = self.config.get("mc_confidence_levels", [0.05, 0.50, 0.95])
        rng = np.random.RandomState(seed)

        if daily_returns.empty or len(daily_returns) < 2:
            return MonteCarloResult()

        arr = daily_returns.values
        n = len(arr)
        sharpes = np.empty(n_sim)
        total_rets = np.empty(n_sim)
        max_dds = np.empty(n_sim)

        for i in range(n_sim):
            sample = rng.choice(arr, size=n, replace=True)
            sample_series = pd.Series(sample)
            sharpes[i] = compute_sharpe(sample_series)
            cum = (1 + sample_series).cumprod()
            total_rets[i] = float(cum.iloc[-1] - 1)
            max_dds[i] = compute_max_drawdown(cum)

        result = MonteCarloResult(
            simulated_sharpes=sharpes,
            simulated_returns=total_rets,
            simulated_max_dds=max_dds,
        )
        for level in ci_levels:
            result.sharpe_ci[level] = float(np.percentile(sharpes, level * 100))
            result.returns_ci[level] = float(np.percentile(total_rets, level * 100))
            result.max_dd_ci[level] = float(np.percentile(max_dds, level * 100))

        return result

    # -- regime metrics -------------------------------------------------------

    def _compute_regime_metrics(
        self, daily_returns: pd.Series, regime_series: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        common_idx = daily_returns.index.intersection(regime_series.index)
        if common_idx.empty:
            return {}
        rets = daily_returns.reindex(common_idx)
        regimes = regime_series.reindex(common_idx)
        result: Dict[str, Dict[str, Any]] = {}
        for regime in regimes.unique():
            mask = regimes == regime
            r = rets[mask]
            eq = (1 + r).cumprod()
            result[str(regime)] = {
                "count": int(mask.sum()),
                "mean_return": float(r.mean()),
                "std_return": float(r.std()) if len(r) > 1 else 0.0,
                "sharpe": compute_sharpe(r),
                "sortino": compute_sortino(r),
                "max_drawdown": compute_max_drawdown(eq),
                "total_return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
            }
        return result

    # -- aggregate metrics ----------------------------------------------------

    @staticmethod
    def _compute_metrics(
        equity: pd.Series, trades: List[Trade], daily_pnl: pd.DataFrame
    ) -> Dict[str, Any]:
        if equity.empty:
            return {}
        returns = equity.pct_change().dropna()
        total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        max_dd = compute_max_drawdown(equity)
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        calmar = compute_calmar(total_return, max_dd)
        win_trades = [t for t in trades if t.pnl > 0]
        lose_trades = [t for t in trades if t.pnl <= 0]
        total_costs = sum(t.costs for t in trades)
        return {
            "total_return": total_return,
            "annualised_return": total_return * (252 / max(len(returns), 1)),
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "num_trades": len(trades),
            "win_rate": len(win_trades) / max(len(trades), 1),
            "avg_trade_pnl": float(np.mean([t.pnl for t in trades]))
            if trades
            else 0.0,
            "total_costs": total_costs,
            "profit_factor": (
                sum(t.pnl for t in win_trades) / abs(sum(t.pnl for t in lose_trades))
                if lose_trades and sum(t.pnl for t in lose_trades) != 0
                else 0.0
            ),
        }

    # -- main run -------------------------------------------------------------

    def run(self) -> BacktestResult:
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        if self._signal_fn is None:
            raise RuntimeError("No signal function set. Call set_signal_fn() first.")

        data = self._data.copy()

        # Feature engineering
        if self._feature_fn is not None:
            data = self._feature_fn(data)

        # Signal generation
        signals = self._signal_fn(data)

        # Core simulation
        equity, trades, pnl_df = self._run_simulation(data, signals)

        # Metrics
        metrics = self._compute_metrics(equity, trades, pnl_df)

        # Walk‑forward
        wf = self._walk_forward(data, signals)

        # Daily returns for MC & regime
        daily_returns = equity.pct_change().dropna()

        # Monte Carlo
        mc = self._monte_carlo(daily_returns)

        # Regime metrics
        regime_metrics: Dict[str, Dict[str, Any]] = {}
        if self._regime_series is not None:
            regime_metrics = self._compute_regime_metrics(
                daily_returns, self._regime_series
            )

        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            daily_pnl=pnl_df,
            metrics=metrics,
            walk_forward_results=wf,
            monte_carlo=mc,
            regime_metrics=regime_metrics,
        )

    # -- HTML report ----------------------------------------------------------

    def generate_report(self, result: BacktestResult) -> str:
        """Return a self‑contained HTML report string."""
        equity = result.equity_curve
        metrics = result.metrics
        trades = result.trades
        pnl = result.daily_pnl

        # SVG equity curve
        eq_svg = self._svg_line_chart(
            equity, title="Equity Curve", width=800, height=300
        )

        # SVG drawdown chart
        dd_series = self._drawdown_series(equity)
        dd_svg = self._svg_line_chart(
            dd_series, title="Drawdown", width=800, height=200, color="red"
        )

        # Monthly returns table
        monthly_html = self._monthly_returns_table(equity)

        # Trade list
        trade_html = self._trade_table(trades)

        # Regime overlay table
        regime_html = self._regime_table(result.regime_metrics)

        # Walk‑forward table
        wf_html = self._walk_forward_table(result.walk_forward_results)

        # MC confidence intervals
        mc_html = self._mc_table(result.monte_carlo)

        # Metrics summary
        metrics_html = self._metrics_table(metrics)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Systematic Backtest Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }}
table {{ border-collapse: collapse; margin-bottom: 20px; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #34495e; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.chart {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1>Systematic Backtest Report</h1>

<h2>Performance Metrics</h2>
{metrics_html}

<h2>Equity Curve</h2>
<div class="chart">{eq_svg}</div>

<h2>Drawdown</h2>
<div class="chart">{dd_svg}</div>

<h2>Monthly Returns</h2>
{monthly_html}

<h2>Trade List</h2>
{trade_html}

<h2>Walk-Forward Validation</h2>
{wf_html}

<h2>Regime Analysis</h2>
{regime_html}

<h2>Monte Carlo Confidence Intervals</h2>
{mc_html}

</body>
</html>"""
        return html

    # -- SVG helpers ----------------------------------------------------------

    @staticmethod
    def _svg_line_chart(
        series: pd.Series,
        title: str = "",
        width: int = 800,
        height: int = 300,
        color: str = "#2980b9",
    ) -> str:
        if series.empty:
            return "<p>No data</p>"
        values = series.values.astype(float)
        n = len(values)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if vmax == vmin:
            vmax = vmin + 1
        margin = 40
        pw = width - 2 * margin
        ph = height - 2 * margin
        points = []
        for i, v in enumerate(values):
            x = margin + (i / max(n - 1, 1)) * pw
            y = margin + ph - ((v - vmin) / (vmax - vmin)) * ph
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        title_esc = html_mod.escape(title)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="14">'
            f"{title_esc}</text>"
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" '
            f'points="{polyline}"/>'
            f"</svg>"
        )
        return svg

    @staticmethod
    def _drawdown_series(equity: pd.Series) -> pd.Series:
        if equity.empty:
            return pd.Series(dtype=float)
        peak = equity.cummax()
        return (equity - peak) / peak.replace(0, np.nan)

    @staticmethod
    def _monthly_returns_table(equity: pd.Series) -> str:
        if equity.empty:
            return "<p>No data</p>"
        monthly = equity.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            return "<p>No monthly data</p>"
        rows_by_year: Dict[int, Dict[int, float]] = {}
        for dt, val in monthly.items():
            y = dt.year  # type: ignore[union-attr]
            m = dt.month  # type: ignore[union-attr]
            rows_by_year.setdefault(y, {})[m] = float(val)

        header = "<tr><th>Year</th>" + "".join(
            f"<th>{m}</th>" for m in range(1, 13)
        ) + "<th>Total</th></tr>"
        body = ""
        for y in sorted(rows_by_year):
            row = f"<tr><td>{y}</td>"
            total = 1.0
            for m in range(1, 13):
                v = rows_by_year[y].get(m)
                if v is not None:
                    row += f"<td>{v:.2%}</td>"
                    total *= 1 + v
                else:
                    row += "<td></td>"
            row += f"<td>{total - 1:.2%}</td></tr>"
            body += row
        return f"<table>{header}{body}</table>"

    @staticmethod
    def _trade_table(trades: List[Trade]) -> str:
        if not trades:
            return "<p>No trades</p>"
        header = (
            "<tr><th>#</th><th>Entry</th><th>Exit</th><th>Dir</th>"
            "<th>Size</th><th>Entry Px</th><th>Exit Px</th>"
            "<th>PnL</th><th>Costs</th><th>Reason</th></tr>"
        )
        rows = ""
        for i, t in enumerate(trades, 1):
            exit_px = f"{t.exit_price:.4f}" if t.exit_price is not None else ""
            entry_dt = t.entry_date.strftime("%Y-%m-%d") if t.entry_date else ""
            exit_dt = t.exit_date.strftime("%Y-%m-%d") if t.exit_date else ""
            rows += (
                f"<tr><td>{i}</td>"
                f"<td>{entry_dt}</td>"
                f"<td>{exit_dt}</td>"
                f"<td>{t.direction}</td>"
                f"<td>{t.size:.2f}</td>"
                f"<td>{t.entry_price:.4f}</td>"
                f"<td>{exit_px}</td>"
                f"<td>{t.pnl:.2f}</td>"
                f"<td>{t.costs:.2f}</td>"
                f"<td>{t.exit_reason}</td></tr>"
            )
        return f"<table>{header}{rows}</table>"

    @staticmethod
    def _regime_table(regime_metrics: Dict[str, Dict[str, Any]]) -> str:
        if not regime_metrics:
            return "<p>No regime data</p>"
        header = (
            "<tr><th>Regime</th><th>Count</th><th>Mean Ret</th>"
            "<th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Total Ret</th></tr>"
        )
        rows = ""
        for regime, m in sorted(regime_metrics.items()):
            rows += (
                f"<tr><td>{html_mod.escape(regime)}</td>"
                f"<td>{m['count']}</td>"
                f"<td>{m['mean_return']:.4%}</td>"
                f"<td>{m['sharpe']:.2f}</td>"
                f"<td>{m['sortino']:.2f}</td>"
                f"<td>{m['max_drawdown']:.2%}</td>"
                f"<td>{m['total_return']:.2%}</td></tr>"
            )
        return f"<table>{header}{rows}</table>"

    @staticmethod
    def _walk_forward_table(wf_results: List[WalkForwardResult]) -> str:
        if not wf_results:
            return "<p>No walk-forward results</p>"
        header = (
            "<tr><th>Split</th><th>Train Start</th><th>Train End</th>"
            "<th>Test Start</th><th>Test End</th>"
            "<th>IS Sharpe</th><th>OOS Sharpe</th>"
            "<th>IS Return</th><th>OOS Return</th></tr>"
        )
        rows = ""
        for wf in wf_results:
            rows += (
                f"<tr><td>{wf.split_index}</td>"
                f"<td>{wf.train_start.strftime('%Y-%m-%d')}</td>"
                f"<td>{wf.train_end.strftime('%Y-%m-%d')}</td>"
                f"<td>{wf.test_start.strftime('%Y-%m-%d')}</td>"
                f"<td>{wf.test_end.strftime('%Y-%m-%d')}</td>"
                f"<td>{wf.in_sample_sharpe:.2f}</td>"
                f"<td>{wf.out_of_sample_sharpe:.2f}</td>"
                f"<td>{wf.in_sample_return:.2%}</td>"
                f"<td>{wf.out_of_sample_return:.2%}</td></tr>"
            )
        return f"<table>{header}{rows}</table>"

    @staticmethod
    def _mc_table(mc: MonteCarloResult) -> str:
        if not mc.sharpe_ci:
            return "<p>No Monte Carlo data</p>"
        header = "<tr><th>Percentile</th><th>Sharpe</th><th>Return</th><th>Max DD</th></tr>"
        rows = ""
        for level in sorted(mc.sharpe_ci.keys()):
            rows += (
                f"<tr><td>{level:.0%}</td>"
                f"<td>{mc.sharpe_ci[level]:.2f}</td>"
                f"<td>{mc.returns_ci[level]:.2%}</td>"
                f"<td>{mc.max_dd_ci[level]:.2%}</td></tr>"
            )
        return f"<table>{header}{rows}</table>"

    @staticmethod
    def _metrics_table(metrics: Dict[str, Any]) -> str:
        if not metrics:
            return "<p>No metrics</p>"
        header = "<tr><th>Metric</th><th>Value</th></tr>"
        rows = ""
        fmt_pct = {"total_return", "annualised_return", "max_drawdown", "win_rate"}
        for k, v in metrics.items():
            if k in fmt_pct:
                rows += f"<tr><td>{k}</td><td>{v:.2%}</td></tr>"
            elif isinstance(v, float):
                rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            else:
                rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
        return f"<table>{header}{rows}</table>"
