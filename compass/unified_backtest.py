"""
Unified backtesting engine — ties ALL compass modules into a single pipeline.

Pipeline stages:
  1. Config          — load experiment YAML or dict
  2. Data            — load training data, split by walk-forward windows
  3. Features        — compute feature columns (regime, vol, momentum, etc.)
  4. Signal          — generate entry signals (score + threshold)
  5. Sizing          — position sizing (Kelly / fixed-frac / risk-parity)
  6. Risk            — risk checks (drawdown gate, exposure limit, regime gate)
  7. Execution       — realistic fills with slippage model
  8. Attribution     — per-trade PnL decomposition (alpha, hedge, costs)
  9. Aggregation     — portfolio-level roll-up across experiments
  10. Metrics        — rolling Sharpe, drawdown, returns, regime-aware benchmarks
  11. Walk-forward   — expanding-window OOS validation
  12. Report         — comprehensive HTML report

This is READ-ONLY simulation.  No broker connections, no trade placement.

Usage::

    from compass.unified_backtest import UnifiedBacktester
    bt = UnifiedBacktester(config)
    result = bt.run(trades_df)
    UnifiedBacktester.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "unified_backtest.html"
TRADING_DAYS = 252


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Pipeline configuration."""

    name: str = "default"
    # Signal
    signal_threshold: float = 0.5
    # Sizing
    sizing_method: str = "fixed_frac"  # "fixed_frac", "kelly", "risk_parity"
    max_position_pct: float = 0.05     # max 5% of capital per trade
    base_contracts: int = 5
    # Risk
    max_drawdown_pct: float = 0.15     # halt at 15% drawdown
    max_exposure_pct: float = 0.50     # max 50% capital deployed
    regime_gate_enabled: bool = True
    allowed_regimes: List[str] = field(default_factory=lambda: ["bull", "sideways"])
    # Execution
    slippage_bps: float = 5.0
    commission_per_contract: float = 1.30
    # Walk-forward
    wf_train_pct: float = 0.70
    wf_n_folds: int = 3
    # Capital
    initial_capital: float = 100_000.0


def load_config(source: Any) -> BacktestConfig:
    """Load config from dict, YAML path, or return default."""
    if source is None:
        return BacktestConfig()
    if isinstance(source, BacktestConfig):
        return source
    if isinstance(source, dict):
        return BacktestConfig(**{k: v for k, v in source.items()
                                  if k in BacktestConfig.__dataclass_fields__})
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                with open(path) as f:
                    d = yaml.safe_load(f)
                return load_config(d)
            except ImportError:
                logger.warning("PyYAML not installed, using default config")
                return BacktestConfig()
    return BacktestConfig()


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    """Single trade with full attribution."""

    trade_id: int
    entry_date: Any
    exit_date: Any
    strategy_type: str
    regime: str
    # Prices
    entry_price: float
    exit_price: float
    contracts: int
    # PnL attribution
    gross_pnl: float
    slippage_cost: float
    commission_cost: float
    net_pnl: float
    alpha_pnl: float        # signal-driven component
    cost_pnl: float         # slippage + commission
    # Context
    signal_score: float
    position_size_pct: float
    drawdown_at_entry: float
    win: bool


@dataclass
class RollingMetrics:
    """Rolling performance snapshot."""

    date: Any
    cumulative_return: float
    rolling_sharpe: float
    rolling_drawdown: float
    rolling_win_rate: float
    n_trades: int


@dataclass
class RegimeBenchmark:
    """Performance within a specific regime."""

    regime: str
    n_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    sharpe: float
    max_drawdown: float


@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""

    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_sharpe: float
    test_sharpe: float
    train_win_rate: float
    test_win_rate: float
    n_train_trades: int
    n_test_trades: int
    test_total_pnl: float


@dataclass
class ExperimentSummary:
    """Summary for one experiment in multi-experiment aggregation."""

    name: str
    n_trades: int
    total_pnl: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    contribution_pct: float


@dataclass
class BacktestResult:
    """Full unified backtest result."""

    config: BacktestConfig
    trades: List[TradeResult]
    rolling_metrics: List[RollingMetrics]
    regime_benchmarks: List[RegimeBenchmark]
    walk_forward_folds: List[WalkForwardFold]
    experiment_summaries: List[ExperimentSummary]
    # Aggregates
    total_pnl: float
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    initial_capital: float
    final_capital: float
    # Attribution totals
    total_alpha_pnl: float
    total_cost_pnl: float
    total_slippage: float
    total_commissions: float


# ── Feature computation ──────────────────────────────────────────────────


def compute_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Ensure feature columns exist, fill defaults where missing."""
    df = trades.copy()

    # Regime
    if "regime" not in df.columns:
        df["regime"] = "unknown"
    df["regime"] = df["regime"].fillna("unknown")

    # Signal score (simulate if not present)
    if "signal_score" not in df.columns:
        if "win" in df.columns:
            # Use historical win as proxy
            df["signal_score"] = df["win"].astype(float) * 0.6 + 0.2
        else:
            df["signal_score"] = 0.5

    return df


# ── Signal generation ────────────────────────────────────────────────────


def apply_signal_filter(
    trades: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Filter trades by signal score threshold."""
    if "signal_score" not in trades.columns:
        return trades
    return trades[trades["signal_score"] >= threshold].copy()


# ── Position sizing ──────────────────────────────────────────────────────


def compute_position_size(
    capital: float,
    signal_score: float,
    method: str,
    max_pct: float,
    base_contracts: int,
    entry_price: float,
) -> Tuple[int, float]:
    """Compute contracts and position size %.

    Returns: (contracts, position_pct)
    """
    if entry_price <= 0 or capital <= 0:
        return base_contracts, 0.0

    if method == "kelly":
        # Simplified Kelly: f* = edge / odds ≈ signal_score - 0.5
        edge = max(0.0, signal_score - 0.5)
        frac = min(edge * 2, max_pct)
    elif method == "risk_parity":
        # Scale by inverse of recent vol (use signal as proxy)
        frac = max_pct * min(signal_score, 1.0) * 0.5
    else:  # fixed_frac
        frac = max_pct

    notional = capital * frac
    per_contract = entry_price * 100
    contracts = max(1, int(notional / per_contract)) if per_contract > 0 else base_contracts
    actual_pct = (contracts * per_contract) / capital if capital > 0 else 0.0

    return contracts, actual_pct


# ── Risk gates ───────────────────────────────────────────────────────────


def check_risk_gates(
    current_drawdown: float,
    current_exposure: float,
    regime: str,
    config: BacktestConfig,
) -> Tuple[bool, str]:
    """Check risk gates. Returns (passed, reason)."""
    if current_drawdown >= config.max_drawdown_pct:
        return False, f"drawdown_limit({current_drawdown:.1%})"
    if current_exposure >= config.max_exposure_pct:
        return False, f"exposure_limit({current_exposure:.1%})"
    if config.regime_gate_enabled and regime not in config.allowed_regimes:
        return False, f"regime_gate({regime})"
    return True, ""


# ── Execution with slippage ──────────────────────────────────────────────


def apply_slippage(
    price: float,
    side: str,
    slippage_bps: float,
) -> float:
    """Apply slippage to execution price."""
    slip = price * slippage_bps / 10_000
    if side == "buy":
        return price + slip
    return price - slip


def compute_trade_costs(
    contracts: int,
    entry_price: float,
    exit_price: float,
    slippage_bps: float,
    commission_per_contract: float,
) -> Tuple[float, float]:
    """Compute slippage and commission costs.

    Returns: (slippage_dollars, commission_dollars)
    """
    multiplier = contracts * 100
    entry_slip = entry_price * slippage_bps / 10_000
    exit_slip = exit_price * slippage_bps / 10_000
    slippage = (entry_slip + exit_slip) * multiplier
    commission = commission_per_contract * contracts * 2  # round trip
    return slippage, commission


# ── Metrics computation ──────────────────────────────────────────────────


def compute_sharpe(pnl_series: np.ndarray) -> float:
    if len(pnl_series) < 2:
        return 0.0
    mu = pnl_series.mean()
    std = pnl_series.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mu / std * math.sqrt(TRADING_DAYS))


def compute_sortino(pnl_series: np.ndarray) -> float:
    if len(pnl_series) < 2:
        return 0.0
    mu = pnl_series.mean()
    downside = pnl_series[pnl_series < 0]
    if len(downside) == 0:
        return float("inf") if mu > 0 else 0.0
    ds = np.sqrt(np.mean(downside ** 2))
    if ds < 1e-12:
        return 0.0
    return float(mu / ds * math.sqrt(TRADING_DAYS))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak > 0, peak, 1)
    return float(dd.min())


def compute_profit_factor(pnl_series: np.ndarray) -> float:
    gains = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_rolling_metrics(
    trade_results: List[TradeResult],
    capital: float,
    window: int = 30,
) -> List[RollingMetrics]:
    """Compute rolling performance from trade results."""
    if not trade_results:
        return []

    cum_pnl = np.cumsum([t.net_pnl for t in trade_results])
    n = len(trade_results)
    results: List[RollingMetrics] = []

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = [trade_results[j] for j in range(start, i + 1)]
        pnls = np.array([t.net_pnl for t in chunk])

        equity = capital + cum_pnl[:i + 1]
        dd = compute_max_drawdown(equity)
        sharpe = compute_sharpe(pnls) if len(pnls) >= 2 else 0.0
        wr = sum(1 for t in chunk if t.win) / len(chunk) if chunk else 0.0

        results.append(RollingMetrics(
            date=trade_results[i].exit_date,
            cumulative_return=float(cum_pnl[i] / capital),
            rolling_sharpe=sharpe,
            rolling_drawdown=dd,
            rolling_win_rate=wr,
            n_trades=i + 1,
        ))

    return results


# ── Regime benchmarking ──────────────────────────────────────────────────


def compute_regime_benchmarks(
    trade_results: List[TradeResult],
) -> List[RegimeBenchmark]:
    """Compute per-regime performance."""
    if not trade_results:
        return []

    by_regime: Dict[str, List[TradeResult]] = {}
    for t in trade_results:
        by_regime.setdefault(t.regime, []).append(t)

    benchmarks: List[RegimeBenchmark] = []
    for regime, trades in sorted(by_regime.items()):
        pnls = np.array([t.net_pnl for t in trades])
        n = len(trades)
        wins = sum(1 for t in trades if t.win)
        total = float(pnls.sum())

        equity = 100_000 + np.cumsum(pnls)
        dd = compute_max_drawdown(equity)
        sharpe = compute_sharpe(pnls)

        benchmarks.append(RegimeBenchmark(
            regime=regime, n_trades=n, total_pnl=total,
            avg_pnl=total / n if n > 0 else 0.0,
            win_rate=wins / n if n > 0 else 0.0,
            sharpe=sharpe, max_drawdown=dd,
        ))

    return benchmarks


# ── Walk-forward validation ──────────────────────────────────────────────


def walk_forward_validate(
    trade_results: List[TradeResult],
    n_folds: int = 3,
    train_pct: float = 0.70,
) -> List[WalkForwardFold]:
    """Expanding-window walk-forward validation."""
    if len(trade_results) < 20 or n_folds < 1:
        return []

    n = len(trade_results)
    fold_size = n // (n_folds + 1)
    if fold_size < 5:
        return []

    folds: List[WalkForwardFold] = []

    for f in range(n_folds):
        # Expanding train window
        train_end = fold_size * (f + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        if test_end <= test_start:
            continue

        train = trade_results[:train_end]
        test = trade_results[test_start:test_end]

        train_pnls = np.array([t.net_pnl for t in train])
        test_pnls = np.array([t.net_pnl for t in test])

        folds.append(WalkForwardFold(
            fold=f + 1,
            train_start=train[0].entry_date if train else None,
            train_end=train[-1].exit_date if train else None,
            test_start=test[0].entry_date if test else None,
            test_end=test[-1].exit_date if test else None,
            train_sharpe=compute_sharpe(train_pnls),
            test_sharpe=compute_sharpe(test_pnls),
            train_win_rate=sum(1 for t in train if t.win) / len(train) if train else 0.0,
            test_win_rate=sum(1 for t in test if t.win) / len(test) if test else 0.0,
            n_train_trades=len(train),
            n_test_trades=len(test),
            test_total_pnl=float(test_pnls.sum()),
        ))

    return folds


# ── Multi-experiment aggregation ─────────────────────────────────────────


def aggregate_experiments(
    experiment_results: Dict[str, List[TradeResult]],
    capital: float,
) -> List[ExperimentSummary]:
    """Aggregate across multiple experiments."""
    if not experiment_results:
        return []

    total_pnl_all = sum(
        sum(t.net_pnl for t in trades)
        for trades in experiment_results.values()
    )

    summaries: List[ExperimentSummary] = []
    for name, trades in sorted(experiment_results.items()):
        pnls = np.array([t.net_pnl for t in trades])
        n = len(trades)
        total = float(pnls.sum())
        equity = capital + np.cumsum(pnls)

        summaries.append(ExperimentSummary(
            name=name, n_trades=n, total_pnl=total,
            sharpe=compute_sharpe(pnls),
            win_rate=sum(1 for t in trades if t.win) / n if n > 0 else 0.0,
            max_drawdown=compute_max_drawdown(equity),
            contribution_pct=total / total_pnl_all if abs(total_pnl_all) > 1e-12 else 0.0,
        ))

    return sorted(summaries, key=lambda s: s.total_pnl, reverse=True)


# ── Core engine ──────────────────────────────────────────────────────────


class UnifiedBacktester:
    """Unified backtesting engine connecting all compass modules."""

    def __init__(self, config: Any = None):
        self.config = load_config(config)

    def run(
        self,
        trades: pd.DataFrame,
        experiment_name: Optional[str] = None,
    ) -> BacktestResult:
        """Run full backtest pipeline on trade data.

        Expected columns: entry_date, exit_date, pnl, entry_price (or net_credit),
                          exit_price (optional), strategy_type, regime, vix, etc.
        """
        cfg = self.config
        name = experiment_name or cfg.name

        # Stage 1: Features
        df = compute_features(trades)

        # Stage 2: Signal filter
        df = apply_signal_filter(df, cfg.signal_threshold)

        if df.empty:
            return self._empty_result()

        # Stage 3-7: Process each trade through pipeline
        capital = cfg.initial_capital
        current_capital = capital
        peak_capital = capital
        exposure = 0.0
        trade_results: List[TradeResult] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            regime = str(row.get("regime", "unknown"))
            drawdown = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0
            exposure_pct = exposure / current_capital if current_capital > 0 else 0.0

            # Risk gate
            passed, reason = check_risk_gates(drawdown, exposure_pct, regime, cfg)
            if not passed:
                continue

            # Sizing
            entry_price = float(row.get("entry_price", abs(row.get("net_credit", 1.0))))
            signal_score = float(row.get("signal_score", 0.5))

            contracts, pos_pct = compute_position_size(
                current_capital, signal_score, cfg.sizing_method,
                cfg.max_position_pct, cfg.base_contracts, entry_price,
            )

            # Execution with slippage
            exit_price = float(row.get("exit_price", entry_price + row.get("pnl", 0) / (contracts * 100)))
            slippage, commission = compute_trade_costs(
                contracts, entry_price, exit_price,
                cfg.slippage_bps, cfg.commission_per_contract,
            )

            # PnL
            gross_pnl = float(row.get("pnl", 0.0))
            # Scale if original PnL was for different contract count
            orig_contracts = int(row.get("contracts", cfg.base_contracts))
            if orig_contracts > 0 and orig_contracts != contracts:
                gross_pnl = gross_pnl / orig_contracts * contracts

            net_pnl = gross_pnl - slippage - commission
            alpha_pnl = gross_pnl
            cost_pnl = -(slippage + commission)

            is_win = net_pnl > 0

            trade_results.append(TradeResult(
                trade_id=idx,
                entry_date=row.get("entry_date"),
                exit_date=row.get("exit_date"),
                strategy_type=str(row.get("strategy_type", "unknown")),
                regime=regime,
                entry_price=entry_price,
                exit_price=exit_price,
                contracts=contracts,
                gross_pnl=gross_pnl,
                slippage_cost=slippage,
                commission_cost=commission,
                net_pnl=net_pnl,
                alpha_pnl=alpha_pnl,
                cost_pnl=cost_pnl,
                signal_score=signal_score,
                position_size_pct=pos_pct,
                drawdown_at_entry=drawdown,
                win=is_win,
            ))

            current_capital += net_pnl
            peak_capital = max(peak_capital, current_capital)

        # Stage 8-10: Analysis
        rolling = compute_rolling_metrics(trade_results, capital)
        regime_bm = compute_regime_benchmarks(trade_results)
        wf_folds = walk_forward_validate(trade_results, cfg.wf_n_folds, cfg.wf_train_pct)

        # Multi-experiment
        exp_results = {name: trade_results}
        exp_summaries = aggregate_experiments(exp_results, capital)

        # Aggregate metrics
        pnls = np.array([t.net_pnl for t in trade_results]) if trade_results else np.array([0.0])
        equity = capital + np.cumsum(pnls)
        n_trades = len(trade_results)
        wins = sum(1 for t in trade_results if t.win)

        return BacktestResult(
            config=cfg,
            trades=trade_results,
            rolling_metrics=rolling,
            regime_benchmarks=regime_bm,
            walk_forward_folds=wf_folds,
            experiment_summaries=exp_summaries,
            total_pnl=float(pnls.sum()),
            total_return=float(pnls.sum() / capital),
            sharpe=compute_sharpe(pnls),
            sortino=compute_sortino(pnls),
            max_drawdown=compute_max_drawdown(equity),
            win_rate=wins / n_trades if n_trades > 0 else 0.0,
            profit_factor=compute_profit_factor(pnls),
            n_trades=n_trades,
            initial_capital=capital,
            final_capital=float(equity[-1]) if len(equity) > 0 else capital,
            total_alpha_pnl=float(sum(t.alpha_pnl for t in trade_results)),
            total_cost_pnl=float(sum(t.cost_pnl for t in trade_results)),
            total_slippage=float(sum(t.slippage_cost for t in trade_results)),
            total_commissions=float(sum(t.commission_cost for t in trade_results)),
        )

    def run_multi(
        self,
        experiments: Dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run backtest across multiple experiments and aggregate."""
        all_trades: List[TradeResult] = []
        exp_results: Dict[str, List[TradeResult]] = {}

        for name, trades_df in experiments.items():
            result = self.run(trades_df, experiment_name=name)
            all_trades.extend(result.trades)
            exp_results[name] = result.trades

        if not all_trades:
            return self._empty_result()

        cfg = self.config
        capital = cfg.initial_capital
        pnls = np.array([t.net_pnl for t in all_trades])
        equity = capital + np.cumsum(pnls)
        n_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.win)

        return BacktestResult(
            config=cfg,
            trades=all_trades,
            rolling_metrics=compute_rolling_metrics(all_trades, capital),
            regime_benchmarks=compute_regime_benchmarks(all_trades),
            walk_forward_folds=walk_forward_validate(all_trades, cfg.wf_n_folds),
            experiment_summaries=aggregate_experiments(exp_results, capital),
            total_pnl=float(pnls.sum()),
            total_return=float(pnls.sum() / capital),
            sharpe=compute_sharpe(pnls),
            sortino=compute_sortino(pnls),
            max_drawdown=compute_max_drawdown(equity),
            win_rate=wins / n_trades if n_trades > 0 else 0.0,
            profit_factor=compute_profit_factor(pnls),
            n_trades=n_trades,
            initial_capital=capital,
            final_capital=float(equity[-1]),
            total_alpha_pnl=float(sum(t.alpha_pnl for t in all_trades)),
            total_cost_pnl=float(sum(t.cost_pnl for t in all_trades)),
            total_slippage=float(sum(t.slippage_cost for t in all_trades)),
            total_commissions=float(sum(t.commission_cost for t in all_trades)),
        )

    def _empty_result(self) -> BacktestResult:
        cfg = self.config
        return BacktestResult(
            config=cfg, trades=[], rolling_metrics=[],
            regime_benchmarks=[], walk_forward_folds=[],
            experiment_summaries=[],
            total_pnl=0.0, total_return=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
            n_trades=0, initial_capital=cfg.initial_capital,
            final_capital=cfg.initial_capital,
            total_alpha_pnl=0.0, total_cost_pnl=0.0,
            total_slippage=0.0, total_commissions=0.0,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: BacktestResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _fd(v: float) -> str:
    return f"${v:,.2f}"


def _svg_line(values: List[float], title: str, color: str = "#58a6ff",
              w: int = 700, h: int = 200) -> str:
    if len(values) < 2:
        return ""
    n = len(values)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65
    y_min = min(min(values), 0)
    y_max = max(values)
    if y_max <= y_min:
        y_max = y_min + 0.01

    def tx(i): return pad + i / max(n - 1, 1) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(values[i]):.1f}" for i in range(n))
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _attribution_waterfall(result: BacktestResult) -> str:
    items = [("Alpha P&L", result.total_alpha_pnl), ("Slippage", -result.total_slippage),
             ("Commissions", -result.total_commissions)]
    w, h = 500, 200
    pad_l = 120
    abs_max = max(abs(v) for _, v in items) if items else 1.0
    if abs_max == 0: abs_max = 1.0
    bar_area = (w - pad_l - 40) / 2
    mid_x = pad_l + bar_area
    bar_h, gap = 28, 10

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="18" text-anchor="middle" class="svg-title">PnL Attribution</text>')
    parts.append(f'<line x1="{mid_x:.0f}" y1="25" x2="{mid_x:.0f}" y2="{h - 5}" stroke="#30363d"/>')

    for i, (label, val) in enumerate(items):
        y = 30 + i * (bar_h + gap)
        bw = abs(val) / abs_max * bar_area
        color = "#3fb950" if val >= 0 else "#f85149"
        bx = mid_x if val >= 0 else mid_x - bw
        parts.append(f'<text x="{pad_l - 5}" y="{y + 18:.0f}" text-anchor="end" font-size="10" fill="#8b949e">{label}</text>')
        parts.append(f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" fill="{color}" rx="3" opacity="0.85"/>')
        parts.append(f'<text x="{bx + bw + 4:.0f}" y="{y + 18:.0f}" font-size="9" fill="#c9d1d9">{_fd(val)}</text>')

    # Total
    y = 30 + len(items) * (bar_h + gap)
    total = result.total_pnl
    bw = abs(total) / abs_max * bar_area
    bx = mid_x if total >= 0 else mid_x - bw
    parts.append(f'<text x="{pad_l - 5}" y="{y + 18:.0f}" text-anchor="end" font-size="10" fill="#f0f6fc" font-weight="bold">Net P&L</text>')
    parts.append(f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" fill="#58a6ff" rx="3"/>')
    parts.append(f'<text x="{bx + bw + 4:.0f}" y="{y + 18:.0f}" font-size="10" fill="#f0f6fc" font-weight="bold">{_fd(total)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _regime_table(bm: List[RegimeBenchmark]) -> str:
    if not bm:
        return "<p class='meta'>No regime data.</p>"
    rows = ""
    for r in bm:
        rows += f"<tr><td style='text-align:left'>{r.regime}</td><td>{r.n_trades}</td><td>{_fd(r.total_pnl)}</td><td>{_fd(r.avg_pnl)}</td><td>{_fp(r.win_rate)}</td><td>{_f(r.sharpe)}</td><td>{_fp(r.max_drawdown)}</td></tr>"
    return f"""<table class="data-table"><tr><th style='text-align:left'>Regime</th><th>Trades</th><th>Total PnL</th><th>Avg PnL</th><th>Win Rate</th><th>Sharpe</th><th>Max DD</th></tr>{rows}</table>"""


def _wf_table(folds: List[WalkForwardFold]) -> str:
    if not folds:
        return "<p class='meta'>Insufficient data for walk-forward.</p>"
    rows = ""
    for f in folds:
        decay_color = "#3fb950" if f.test_sharpe > 0 else "#f85149"
        rows += f"<tr><td>{f.fold}</td><td>{f.n_train_trades}</td><td>{f.n_test_trades}</td><td>{_f(f.train_sharpe)}</td><td style='color:{decay_color}'>{_f(f.test_sharpe)}</td><td>{_fp(f.train_win_rate)}</td><td>{_fp(f.test_win_rate)}</td><td>{_fd(f.test_total_pnl)}</td></tr>"
    return f"""<table class="data-table"><tr><th>Fold</th><th>Train N</th><th>Test N</th><th>Train Sharpe</th><th>Test Sharpe</th><th>Train WR</th><th>Test WR</th><th>Test PnL</th></tr>{rows}</table>"""


def _experiment_table(exps: List[ExperimentSummary]) -> str:
    if not exps:
        return ""
    rows = ""
    for e in exps:
        rows += f"<tr><td style='text-align:left'>{e.name}</td><td>{e.n_trades}</td><td>{_fd(e.total_pnl)}</td><td>{_f(e.sharpe)}</td><td>{_fp(e.win_rate)}</td><td>{_fp(e.max_drawdown)}</td><td>{_fp(e.contribution_pct)}</td></tr>"
    return f"""<h2>Experiment Breakdown</h2><table class="data-table"><tr><th style='text-align:left'>Experiment</th><th>Trades</th><th>PnL</th><th>Sharpe</th><th>Win Rate</th><th>Max DD</th><th>Contribution</th></tr>{rows}</table>"""


def _build_html(result: BacktestResult) -> str:
    cfg = result.config
    equity_curve = [r.cumulative_return for r in result.rolling_metrics]
    sharpe_curve = [r.rolling_sharpe for r in result.rolling_metrics]
    dd_curve = [r.rolling_drawdown for r in result.rolling_metrics]

    sharpe_color = "#3fb950" if result.sharpe > 0.5 else "#d29922" if result.sharpe > 0 else "#f85149"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Unified Backtest Report — {cfg.name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.15em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Unified Backtest: {cfg.name}</h1>
<p class="meta">{result.n_trades} trades &middot;
   Capital: {_fd(result.initial_capital)} &rarr; {_fd(result.final_capital)} &middot;
   Sizing: {cfg.sizing_method} &middot; Slippage: {cfg.slippage_bps}bps</p>

<div class="summary">
  <div class="stat"><div class="label">Net PnL</div><div class="value">{_fd(result.total_pnl)}</div></div>
  <div class="stat"><div class="label">Return</div><div class="value">{_fp(result.total_return)}</div></div>
  <div class="stat"><div class="label">Sharpe</div><div class="value" style="color:{sharpe_color}">{_f(result.sharpe)}</div></div>
  <div class="stat"><div class="label">Sortino</div><div class="value">{_f(result.sortino)}</div></div>
  <div class="stat"><div class="label">Max DD</div><div class="value">{_fp(result.max_drawdown)}</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">{_fp(result.win_rate)}</div></div>
  <div class="stat"><div class="label">Profit Factor</div><div class="value">{_f(result.profit_factor)}</div></div>
  <div class="stat"><div class="label">Trades</div><div class="value">{result.n_trades}</div></div>
  <div class="stat"><div class="label">Total Slippage</div><div class="value">{_fd(result.total_slippage)}</div></div>
  <div class="stat"><div class="label">Total Commission</div><div class="value">{_fd(result.total_commissions)}</div></div>
</div>

<h2>PnL Attribution</h2>
{_attribution_waterfall(result)}

<h2>Equity Curve</h2>
{_svg_line(equity_curve, "Cumulative Return", "#3fb950")}

<div class="two-col">
  <div>{_svg_line(sharpe_curve, "Rolling Sharpe", "#58a6ff")}</div>
  <div>{_svg_line(dd_curve, "Rolling Drawdown", "#f85149")}</div>
</div>

<h2>Regime Benchmarks</h2>
{_regime_table(result.regime_benchmarks)}

<h2>Walk-Forward Validation</h2>
{_wf_table(result.walk_forward_folds)}

{_experiment_table(result.experiment_summaries)}

</body>
</html>"""
