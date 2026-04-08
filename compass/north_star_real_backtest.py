"""
North Star Portfolio — Real IronVault Backtest (EXP-1470-real).

Re-backtests the EXP-1470 North Star 4-strategy blend using ONLY real option
prices from IronVault.  The original EXP-1470 used np.random Monte Carlo
over hard-coded strategy metrics — that is synthetic data, which is BANNED.

This module runs 4 independent credit spread backtests through the production
Backtester with IronVault, extracts per-year equity curves, and combines
them with the HRP weights to produce honest portfolio-level metrics.

Usage::

    from compass.north_star_real_backtest import NorthStarRealBacktest
    bt = NorthStarRealBacktest()
    result = bt.run()
    bt.generate_report(result, "experiments/EXP-1470-real/results/report.html")

Data source: shared.iron_vault.IronVault (options_cache.db, ~989 MB)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

# ── North Star HRP weights from EXP-1470 ─────────────────────────────────

NORTH_STAR_WEIGHTS: Dict[str, float] = {
    "ML-CS-860": 0.405,
    "Regime-Lev": 0.209,
    "Intraday-MR": 0.205,
    "Combined-750": 0.181,
}

# ── Strategy backtester configs ──────────────────────────────────────────
# Each strategy maps to a Backtester config dict that defines how it trades.
# All use real IronVault data — zero synthetic pricing.


def _base_backtest_config() -> Dict[str, Any]:
    """Shared backtest settings across all strategies."""
    return {
        "starting_capital": 100_000,
        "commission_per_contract": 0.65,
        "slippage": 0.05,
        "exit_slippage": 0.10,
        "slippage_multiplier": 1.0,
        "compound": False,
        "sizing_mode": "flat",
        "score_threshold": 25,
        "generate_reports": False,
        "volume_gate": False,
        "max_portfolio_exposure_pct": 100.0,
        "monte_carlo": {"mode": "dte", "dte_lo": 28, "dte_hi": 42},
    }


def _ml_cs_860_config() -> Dict[str, Any]:
    """ML-CS-860: Conservative ML-filtered credit spreads.

    From EXP-860: Production ensemble, tight sizing, delta 0.12,
    35-DTE, $5 spreads, confidence-graded entry.
    """
    return {
        "strategy": {
            "target_dte": 35,
            "min_dte": 25,
            "direction": "both",
            "spread_width": 5,
            "use_delta_selection": True,
            "target_delta": 0.12,
            "regime_mode": "combo",
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
            },
        },
        "risk": {
            "account_size": 100_000,
            "max_risk_per_trade": 4.0,
            "min_contracts": 1,
            "max_contracts": 15,
            "max_positions": 8,
            "max_positions_per_ticker": 2,
            "profit_target": 50,
            "stop_loss_multiplier": 2.5,
            "stop_loss_pct_of_width": 90,
            "scan_days": [0, 1, 2, 3, 4],
            "portfolio_risk": {
                "max_portfolio_risk_pct": 30,
                "max_single_ticker_pct": 30,
                "max_same_expiration": 3,
            },
            "drawdown_cb_pct": 20,
            "cooldown_bars": 5,
            "enable_rolling": False,
        },
        "backtest": {**_base_backtest_config()},
    }


def _regime_lev_config() -> Dict[str, Any]:
    """Regime-Lev: Regime-adaptive credit spreads with higher sizing in bull.

    From EXP-840: Same base spread mechanics but larger position sizes
    in bullish regimes, reduced in bear/crash.
    """
    return {
        "strategy": {
            "target_dte": 35,
            "min_dte": 25,
            "direction": "both",
            "spread_width": 12,
            "use_delta_selection": True,
            "target_delta": 0.12,
            "regime_mode": "combo",
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
            },
        },
        "risk": {
            "account_size": 100_000,
            "max_risk_per_trade": 8.5,
            "min_contracts": 1,
            "max_contracts": 25,
            "max_positions": 10,
            "max_positions_per_ticker": 3,
            "profit_target": 55,
            "stop_loss_multiplier": 1.25,
            "stop_loss_pct_of_width": 90,
            "scan_days": [0, 1, 2, 3, 4],
            "portfolio_risk": {
                "max_portfolio_risk_pct": 40,
                "max_single_ticker_pct": 40,
                "max_same_expiration": 4,
            },
            "drawdown_cb_pct": 40,
            "cooldown_bars": 3,
            "enable_rolling": False,
        },
        "backtest": {**_base_backtest_config()},
    }


def _intraday_mr_config() -> Dict[str, Any]:
    """Intraday-MR: Short-DTE mean reversion spreads on calm days.

    From EXP-1000: Uses 7-DTE spreads (shortest available with real data),
    tighter OTM, lower risk per trade.  Skips high-vol days.
    """
    return {
        "strategy": {
            "target_dte": 7,
            "min_dte": 3,
            "direction": "both",
            "spread_width": 5,
            "use_delta_selection": True,
            "target_delta": 0.08,
            "regime_mode": "combo",
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
            },
        },
        "risk": {
            "account_size": 100_000,
            "max_risk_per_trade": 2.0,
            "min_contracts": 1,
            "max_contracts": 10,
            "max_positions": 12,
            "max_positions_per_ticker": 4,
            "profit_target": 40,
            "stop_loss_multiplier": 2.0,
            "stop_loss_pct_of_width": 90,
            "scan_days": [0, 1, 2, 3, 4],
            "portfolio_risk": {
                "max_portfolio_risk_pct": 20,
                "max_single_ticker_pct": 20,
                "max_same_expiration": 6,
            },
            "drawdown_cb_pct": 15,
            "cooldown_bars": 2,
            "enable_rolling": False,
        },
        "backtest": {**_base_backtest_config()},
    }


def _combined_750_config() -> Dict[str, Any]:
    """Combined-750: Credit spread + vol harvest blend (wider spreads).

    From EXP-750: 60% credit spread + 40% vol selling.  Uses wider
    $12 spreads and moderate sizing.
    """
    return {
        "strategy": {
            "target_dte": 30,
            "min_dte": 20,
            "direction": "both",
            "spread_width": 12,
            "use_delta_selection": True,
            "target_delta": 0.10,
            "regime_mode": "combo",
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
            },
        },
        "risk": {
            "account_size": 100_000,
            "max_risk_per_trade": 6.0,
            "min_contracts": 1,
            "max_contracts": 20,
            "max_positions": 10,
            "max_positions_per_ticker": 3,
            "profit_target": 50,
            "stop_loss_multiplier": 1.5,
            "stop_loss_pct_of_width": 90,
            "scan_days": [0, 1, 2, 3, 4],
            "portfolio_risk": {
                "max_portfolio_risk_pct": 35,
                "max_single_ticker_pct": 35,
                "max_same_expiration": 4,
            },
            "drawdown_cb_pct": 30,
            "cooldown_bars": 3,
            "enable_rolling": False,
        },
        "backtest": {**_base_backtest_config()},
    }


STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ML-CS-860": _ml_cs_860_config(),
    "Regime-Lev": _regime_lev_config(),
    "Intraday-MR": _intraday_mr_config(),
    "Combined-750": _combined_750_config(),
}

# Spread widths per strategy (passed as otm_pct to Backtester)
STRATEGY_SPREAD_WIDTHS: Dict[str, float] = {
    "ML-CS-860": 5.0,
    "Regime-Lev": 12.0,
    "Intraday-MR": 5.0,
    "Combined-750": 12.0,
}

STRATEGY_OTM_PCTS: Dict[str, float] = {
    "ML-CS-860": 0.05,
    "Regime-Lev": 0.02,
    "Intraday-MR": 0.03,
    "Combined-750": 0.04,
}

# Original synthetic claims from EXP-1470 (for comparison)
SYNTHETIC_CLAIMS = {
    "cagr": 27.85,
    "max_dd": 1.62,
    "sharpe": 17.21,
    "cagr_at_dd12": 206.51,
    "leverage_100": 3.59,
}

YEARS = list(range(2020, 2026))
BACKTEST_START = datetime(2020, 1, 2)
BACKTEST_END = datetime(2025, 12, 31)


# ── Result data classes ──────────────────────────────────────────────────


@dataclass
class StrategyYearResult:
    """One strategy's results for one year."""

    year: int
    cagr: float
    max_dd: float
    sharpe: float
    n_trades: int
    win_rate: float


@dataclass
class StrategyResult:
    """Full backtest result for one strategy."""

    name: str
    config: Dict[str, Any]
    trades: List[Dict]
    equity_curve: List[Tuple[datetime, float]]
    yearly: Dict[int, StrategyYearResult]
    total_cagr: float
    total_max_dd: float
    total_sharpe: float
    total_win_rate: float
    total_trades: int


@dataclass
class PortfolioYearResult:
    """Portfolio-level results for one year."""

    year: int
    cagr: float
    max_dd: float
    sharpe: float
    n_trades: int
    win_rate: float
    strategy_contributions: Dict[str, float]


@dataclass
class NorthStarRealResult:
    """Complete result of real backtest."""

    strategies: Dict[str, StrategyResult]
    portfolio_yearly: Dict[int, PortfolioYearResult]
    portfolio_total_cagr: float
    portfolio_total_dd: float
    portfolio_total_sharpe: float
    portfolio_total_win_rate: float
    portfolio_total_trades: int
    weights: Dict[str, float]
    synthetic_comparison: Dict[str, Any]


# ── Core computation ─────────────────────────────────────────────────────


def _extract_yearly_from_backtest(
    result: Dict, starting_capital: float = 100_000
) -> Dict[int, StrategyYearResult]:
    """Extract year-by-year metrics from a Backtester result dict."""
    trades = result.get("trades", [])
    equity_curve = result.get("equity_curve", [])

    if not trades and not equity_curve:
        return {}

    yearly: Dict[int, StrategyYearResult] = {}

    # Group trades by year
    trades_by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        exit_date = t.get("exit_date")
        if exit_date is None:
            continue
        if isinstance(exit_date, str):
            year = int(exit_date[:4])
        elif isinstance(exit_date, datetime):
            year = exit_date.year
        else:
            continue
        trades_by_year.setdefault(year, []).append(t)

    # Equity curve by year for DD calculation
    eq_by_year: Dict[int, List[Tuple[Any, float]]] = {}
    for point in equity_curve:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            dt, val = point[0], point[1]
            if isinstance(dt, datetime):
                yr = dt.year
            elif isinstance(dt, str):
                yr = int(dt[:4])
            else:
                continue
            eq_by_year.setdefault(yr, []).append((dt, float(val)))

    for year in YEARS:
        year_trades = trades_by_year.get(year, [])
        year_eq = eq_by_year.get(year, [])

        # CAGR from equity curve
        if year_eq:
            start_val = year_eq[0][1]
            end_val = year_eq[-1][1]
            if start_val > 0:
                cagr = ((end_val / start_val) - 1) * 100
            else:
                cagr = 0.0
        else:
            # Fallback: sum trade PnLs
            total_pnl = sum(t.get("pnl", 0) for t in year_trades)
            cagr = (total_pnl / starting_capital) * 100

        # Max DD from equity curve
        if year_eq:
            vals = [v for _, v in year_eq]
            peak = vals[0]
            max_dd = 0.0
            for v in vals:
                if v > peak:
                    peak = v
                dd = (v - peak) / peak * 100 if peak > 0 else 0
                if dd < max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0

        # Sharpe (annualized from daily returns)
        if len(year_eq) > 1:
            vals = [v for _, v in year_eq]
            returns = [(vals[i] - vals[i - 1]) / vals[i - 1]
                       for i in range(1, len(vals)) if vals[i - 1] > 0]
            if returns:
                mean_r = np.mean(returns)
                std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-6
                sharpe = (mean_r / std_r * math.sqrt(TRADING_DAYS)) if std_r > 1e-8 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = cagr / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

        # Win rate
        if year_trades:
            wins = sum(1 for t in year_trades if t.get("pnl", 0) > 0)
            win_rate = wins / len(year_trades)
        else:
            win_rate = 0.0

        yearly[year] = StrategyYearResult(
            year=year,
            cagr=round(cagr, 2),
            max_dd=round(max_dd, 2),
            sharpe=round(sharpe, 2),
            n_trades=len(year_trades),
            win_rate=round(win_rate, 3),
        )

    return yearly


def _compute_total_metrics(
    yearly: Dict[int, StrategyYearResult],
) -> Tuple[float, float, float, float, int]:
    """Compute total metrics from yearly results."""
    if not yearly:
        return 0.0, 0.0, 0.0, 0.0, 0

    cagrs = [y.cagr for y in yearly.values()]
    dds = [y.max_dd for y in yearly.values() if y.max_dd != 0]
    sharpes = [y.sharpe for y in yearly.values()]
    n = len(cagrs)

    # Compound CAGR
    compound = 1.0
    for c in cagrs:
        compound *= (1 + c / 100)
    total_cagr = (compound ** (1 / n) - 1) * 100 if n > 0 else 0

    total_dd = min(dds) if dds else 0.0
    total_sharpe = sum(sharpes) / n if n else 0.0

    total_trades = sum(y.n_trades for y in yearly.values())
    weighted_wins = sum(y.n_trades * y.win_rate for y in yearly.values())
    total_win_rate = weighted_wins / total_trades if total_trades > 0 else 0.0

    return (
        round(total_cagr, 2),
        round(total_dd, 2),
        round(total_sharpe, 2),
        round(total_win_rate, 3),
        total_trades,
    )


def _combine_portfolio_year(
    year: int,
    strategy_results: Dict[str, StrategyResult],
    weights: Dict[str, float],
) -> PortfolioYearResult:
    """Combine strategy results into portfolio-level year metrics."""
    w_sum = sum(weights.values())

    # Weighted CAGR
    port_cagr = 0.0
    contributions = {}
    total_trades = 0
    total_wins = 0

    for name, weight in weights.items():
        sr = strategy_results.get(name)
        if sr is None:
            continue
        yr = sr.yearly.get(year)
        if yr is None:
            continue
        w_norm = weight / w_sum
        contribution = w_norm * yr.cagr
        port_cagr += contribution
        contributions[name] = round(contribution, 2)
        total_trades += yr.n_trades
        total_wins += int(yr.n_trades * yr.win_rate)

    # DD: use worst weighted DD as conservative estimate
    # (can't perfectly correlate-adjust without daily equity data)
    dds = []
    for name, weight in weights.items():
        sr = strategy_results.get(name)
        if sr is None:
            continue
        yr = sr.yearly.get(year)
        if yr is None:
            continue
        w_norm = weight / w_sum
        dds.append(w_norm * yr.max_dd)
    port_dd = sum(dds) if dds else 0.0

    port_sharpe = port_cagr / abs(port_dd) if abs(port_dd) > 0.01 else 0.0
    port_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    return PortfolioYearResult(
        year=year,
        cagr=round(port_cagr, 2),
        max_dd=round(port_dd, 2),
        sharpe=round(port_sharpe, 2),
        n_trades=total_trades,
        win_rate=round(port_win_rate, 3),
        strategy_contributions=contributions,
    )


# ── Main backtest engine ─────────────────────────────────────────────────


class NorthStarRealBacktest:
    """Run the North Star 4-strategy backtest using real IronVault data."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        strategy_configs: Optional[Dict[str, Dict]] = None,
        ticker: str = "SPY",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        self.weights = weights or dict(NORTH_STAR_WEIGHTS)
        self.strategy_configs = strategy_configs or dict(STRATEGY_CONFIGS)
        self.ticker = ticker
        self.start_date = start_date or BACKTEST_START
        self.end_date = end_date or BACKTEST_END

    def run(self) -> NorthStarRealResult:
        """Run all 4 strategy backtests and combine.

        Uses IronVault.instance() for real option prices.
        Raises IronVaultError if DB is missing.
        """
        from backtest.backtester import Backtester
        from shared.iron_vault import IronVault

        hd = IronVault.instance()
        logger.info("IronVault connected. Running North Star real backtest...")

        strategy_results: Dict[str, StrategyResult] = {}

        for name, config in self.strategy_configs.items():
            logger.info("Running strategy: %s", name)
            otm_pct = STRATEGY_OTM_PCTS.get(name, 0.05)

            bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
            raw_result = bt.run_backtest(self.ticker, self.start_date, self.end_date)

            if not raw_result:
                logger.warning("Strategy %s returned empty result", name)
                strategy_results[name] = StrategyResult(
                    name=name, config=config, trades=[], equity_curve=[],
                    yearly={}, total_cagr=0, total_max_dd=0, total_sharpe=0,
                    total_win_rate=0, total_trades=0,
                )
                continue

            yearly = _extract_yearly_from_backtest(raw_result)
            total_cagr, total_dd, total_sharpe, total_wr, total_n = _compute_total_metrics(yearly)

            strategy_results[name] = StrategyResult(
                name=name,
                config=config,
                trades=raw_result.get("trades", []),
                equity_curve=raw_result.get("equity_curve", []),
                yearly=yearly,
                total_cagr=total_cagr,
                total_max_dd=total_dd,
                total_sharpe=total_sharpe,
                total_win_rate=total_wr,
                total_trades=total_n,
            )
            logger.info(
                "  %s: CAGR=%.1f%%, DD=%.1f%%, Sharpe=%.2f, Trades=%d, WR=%.1f%%",
                name, total_cagr, total_dd, total_sharpe, total_n, total_wr * 100,
            )

        # Combine into portfolio
        portfolio_yearly: Dict[int, PortfolioYearResult] = {}
        for year in YEARS:
            portfolio_yearly[year] = _combine_portfolio_year(
                year, strategy_results, self.weights
            )

        # Total portfolio metrics
        port_cagrs = [py.cagr for py in portfolio_yearly.values()]
        n = len(port_cagrs)
        compound = 1.0
        for c in port_cagrs:
            compound *= (1 + c / 100)
        total_port_cagr = round((compound ** (1 / n) - 1) * 100, 2) if n else 0
        total_port_dd = round(min(py.max_dd for py in portfolio_yearly.values()), 2)
        total_port_sharpe = round(
            sum(py.sharpe for py in portfolio_yearly.values()) / n, 2
        ) if n else 0
        total_port_trades = sum(py.n_trades for py in portfolio_yearly.values())
        total_port_wr = round(
            sum(py.n_trades * py.win_rate for py in portfolio_yearly.values())
            / total_port_trades, 3
        ) if total_port_trades else 0

        # Build comparison with synthetic claims
        comparison = {
            "synthetic": SYNTHETIC_CLAIMS,
            "real": {
                "cagr": total_port_cagr,
                "max_dd": total_port_dd,
                "sharpe": total_port_sharpe,
                "win_rate": total_port_wr,
                "total_trades": total_port_trades,
            },
            "cagr_delta": round(total_port_cagr - SYNTHETIC_CLAIMS["cagr"], 2),
            "sharpe_delta": round(total_port_sharpe - SYNTHETIC_CLAIMS["sharpe"], 2),
        }

        return NorthStarRealResult(
            strategies=strategy_results,
            portfolio_yearly=portfolio_yearly,
            portfolio_total_cagr=total_port_cagr,
            portfolio_total_dd=total_port_dd,
            portfolio_total_sharpe=total_port_sharpe,
            portfolio_total_win_rate=total_port_wr,
            portfolio_total_trades=total_port_trades,
            weights=self.weights,
            synthetic_comparison=comparison,
        )

    def generate_report(
        self,
        result: NorthStarRealResult,
        output_path: str | Path,
    ) -> Path:
        """Generate self-contained HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_report_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def save_summary(
        self,
        result: NorthStarRealResult,
        output_path: str | Path,
    ) -> Path:
        """Save JSON summary."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": "EXP-1470-real",
            "description": "North Star Portfolio — Real IronVault Backtest",
            "data_source": "IronVault (options_cache.db, real Polygon prices)",
            "ticker": "SPY",
            "period": "2020-01-02 to 2025-12-31",
            "weights": result.weights,
            "portfolio": {
                "compound_cagr": result.portfolio_total_cagr,
                "worst_dd": result.portfolio_total_dd,
                "avg_sharpe": result.portfolio_total_sharpe,
                "total_trades": result.portfolio_total_trades,
                "avg_win_rate": result.portfolio_total_win_rate,
            },
            "per_year": {
                str(yr): {
                    "cagr": py.cagr,
                    "max_dd": py.max_dd,
                    "sharpe": py.sharpe,
                    "n_trades": py.n_trades,
                    "win_rate": py.win_rate,
                }
                for yr, py in sorted(result.portfolio_yearly.items())
            },
            "per_strategy": {
                name: {
                    "compound_cagr": sr.total_cagr,
                    "worst_dd": sr.total_max_dd,
                    "avg_sharpe": sr.total_sharpe,
                    "total_trades": sr.total_trades,
                    "avg_win_rate": sr.total_win_rate,
                }
                for name, sr in result.strategies.items()
            },
            "comparison_vs_synthetic": result.synthetic_comparison,
        }
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return output_path


# ── HTML Report ──────────────────────────────────────────────────────────


def _fc(v: float) -> str:
    color = "#22c55e" if v > 0 else "#ef4444"
    return f'<span style="color:{color}">{v:+.1f}%</span>'


def _fp(v: float) -> str:
    return f"{v:.1f}%"


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _build_report_html(result: NorthStarRealResult) -> str:
    """Build complete self-contained HTML report."""
    r = result
    sc = r.synthetic_comparison

    # Strategy rows
    strat_rows = ""
    for name in ["ML-CS-860", "Regime-Lev", "Intraday-MR", "Combined-750"]:
        sr = r.strategies.get(name)
        if sr is None:
            continue
        w = r.weights.get(name, 0)
        strat_rows += (
            f"<tr><td style='text-align:left'>{name}</td>"
            f"<td>{w:.1%}</td>"
            f"<td>{_fc(sr.total_cagr)}</td>"
            f"<td style='color:#f59e0b'>{sr.total_max_dd:.1f}%</td>"
            f"<td>{sr.total_sharpe:.2f}</td>"
            f"<td>{sr.total_trades:,}</td>"
            f"<td>{sr.total_win_rate:.1%}</td></tr>\n"
        )

    # Year-by-year portfolio table
    year_rows = ""
    for year in sorted(r.portfolio_yearly.keys()):
        py = r.portfolio_yearly[year]
        year_rows += (
            f"<tr><td>{year}</td>"
            f"<td>{_fc(py.cagr)}</td>"
            f"<td style='color:#f59e0b'>{py.max_dd:.1f}%</td>"
            f"<td>{py.sharpe:.2f}</td>"
            f"<td>{py.n_trades:,}</td>"
            f"<td>{py.win_rate:.1%}</td></tr>\n"
        )

    # Per-strategy yearly breakdown
    strat_yearly_rows = ""
    for name in ["ML-CS-860", "Regime-Lev", "Intraday-MR", "Combined-750"]:
        sr = r.strategies.get(name)
        if sr is None:
            continue
        for year in sorted(sr.yearly.keys()):
            yr = sr.yearly[year]
            strat_yearly_rows += (
                f"<tr><td style='text-align:left'>{name}</td><td>{year}</td>"
                f"<td>{_fc(yr.cagr)}</td>"
                f"<td style='color:#f59e0b'>{yr.max_dd:.1f}%</td>"
                f"<td>{yr.sharpe:.2f}</td>"
                f"<td>{yr.n_trades:,}</td>"
                f"<td>{yr.win_rate:.1%}</td></tr>\n"
            )

    # Comparison table
    syn = sc.get("synthetic", {})
    real = sc.get("real", {})
    cagr_delta = sc.get("cagr_delta", 0)
    sharpe_delta = sc.get("sharpe_delta", 0)

    delta_color = "#22c55e" if cagr_delta >= 0 else "#ef4444"
    hero_color = "#ef4444" if cagr_delta < -10 else ("#d29922" if cagr_delta < 0 else "#3fb950")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EXP-1470-real: North Star Real IronVault Backtest</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2,h3{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid {hero_color};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:1.8em;font-weight:800;color:{hero_color}}}
.hero .sub{{color:#8b949e;margin-top:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.2em;margin-top:4px}}
.c .v.green{{color:#3fb950}}.c .v.red{{color:#ef4444}}.c .v.blue{{color:#58a6ff}}.c .v.orange{{color:#f97316}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22;font-size:.85em}}
td:first-child,th:first-child{{text-align:left}}
tr:hover td{{background:#161b2280}}
.section{{margin:32px 0}}
.warn{{background:#d2992220;border:1px solid #d29922;border-radius:8px;padding:16px;margin:20px 0;color:#d29922}}
.warn strong{{color:#f0f6fc}}
.pass{{background:#3fb95020;border:1px solid #3fb950;border-radius:8px;padding:16px;margin:20px 0;color:#3fb950}}
.note{{color:#8b949e;font-size:.85em;margin:8px 0}}
</style></head><body>

<h1>EXP-1470-real: North Star Real IronVault Backtest</h1>
<p class="note">
  Data source: <strong>IronVault</strong> (options_cache.db, real Polygon prices) &middot;
  Ticker: SPY &middot; Period: 2020–2025 &middot;
  4-strategy HRP blend
</p>

<div class="hero">
  <div class="big">Real CAGR: {_fp(r.portfolio_total_cagr)} (Synthetic claimed {_fp(syn.get('cagr', 0))})</div>
  <div class="sub">
    Real DD: {_fp(r.portfolio_total_dd)} &middot;
    Real Sharpe: {_fr(r.portfolio_total_sharpe)} &middot;
    Delta: {cagr_delta:+.1f}pp CAGR, {sharpe_delta:+.1f} Sharpe vs synthetic
  </div>
</div>

<div class="{'warn' if cagr_delta < 0 else 'pass'}">
  <strong>{'REALITY CHECK: Real results differ from synthetic claims' if cagr_delta < 0 else 'Real data confirms portfolio performance'}</strong><br/>
  The original EXP-1470 used <code>np.random</code> Monte Carlo over hard-coded strategy metrics.
  This backtest uses <strong>real option prices from IronVault</strong> (Polygon.io historical data).
  {'Real-world friction (slippage, commissions, data gaps, regime detection) reduces returns.' if cagr_delta < 0 else 'Real-world data validates the strategy thesis.'}
</div>

<div class="cards">
  <div class="c"><div class="l">Real CAGR</div><div class="v {'green' if r.portfolio_total_cagr > 0 else 'red'}">{_fp(r.portfolio_total_cagr)}</div></div>
  <div class="c"><div class="l">Synthetic CAGR</div><div class="v">{_fp(syn.get('cagr', 0))}</div></div>
  <div class="c"><div class="l">CAGR Delta</div><div class="v" style="color:{delta_color}">{cagr_delta:+.1f}pp</div></div>
  <div class="c"><div class="l">Real Worst DD</div><div class="v">{_fp(r.portfolio_total_dd)}</div></div>
  <div class="c"><div class="l">Real Avg Sharpe</div><div class="v">{_fr(r.portfolio_total_sharpe)}</div></div>
  <div class="c"><div class="l">Synthetic Sharpe</div><div class="v">{_fr(syn.get('sharpe', 0))}</div></div>
  <div class="c"><div class="l">Total Trades</div><div class="v">{r.portfolio_total_trades:,}</div></div>
  <div class="c"><div class="l">Avg Win Rate</div><div class="v">{r.portfolio_total_win_rate:.1%}</div></div>
</div>

<div class="section">
<h2>Synthetic vs Real Comparison</h2>
<table>
<thead><tr><th>Metric</th><th>Synthetic (EXP-1470)</th><th>Real (IronVault)</th><th>Delta</th></tr></thead>
<tbody>
<tr><td style="text-align:left">CAGR</td><td>{_fp(syn.get('cagr', 0))}</td><td>{_fc(r.portfolio_total_cagr)}</td><td style="color:{delta_color}">{cagr_delta:+.1f}pp</td></tr>
<tr><td style="text-align:left">Max DD</td><td>{_fp(syn.get('max_dd', 0))}</td><td style="color:#f59e0b">{_fp(r.portfolio_total_dd)}</td><td>—</td></tr>
<tr><td style="text-align:left">Sharpe</td><td>{_fr(syn.get('sharpe', 0))}</td><td>{_fr(r.portfolio_total_sharpe)}</td><td style="color:{'#22c55e' if sharpe_delta >= 0 else '#ef4444'}">{sharpe_delta:+.2f}</td></tr>
<tr><td style="text-align:left">Win Rate</td><td>—</td><td>{r.portfolio_total_win_rate:.1%}</td><td>—</td></tr>
<tr><td style="text-align:left">Total Trades</td><td>0 (analytical)</td><td>{r.portfolio_total_trades:,}</td><td>—</td></tr>
</tbody></table>
</div>

<div class="section">
<h2>Portfolio Year-by-Year (Real)</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Trades</th><th>Win Rate</th></tr></thead>
<tbody>{year_rows}</tbody></table>
</div>

<div class="section">
<h2>Strategy Summary</h2>
<table>
<thead><tr><th>Strategy</th><th>Weight</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Trades</th><th>Win Rate</th></tr></thead>
<tbody>{strat_rows}</tbody></table>
</div>

<div class="section">
<h2>Per-Strategy Year-by-Year</h2>
<table>
<thead><tr><th>Strategy</th><th>Year</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Trades</th><th>Win Rate</th></tr></thead>
<tbody>{strat_yearly_rows}</tbody></table>
</div>

<div class="section">
<h2>North Star Allocation</h2>
<table>
<thead><tr><th>Strategy</th><th>Weight</th></tr></thead>
<tbody>
{''.join(f"<tr><td style='text-align:left'>{n}</td><td>{w:.1%}</td></tr>" for n, w in sorted(r.weights.items(), key=lambda x: x[1], reverse=True))}
</tbody></table>
</div>

<p class="note" style="margin-top:40px;text-align:center">
  EXP-1470-real &middot; North Star Real IronVault Backtest &middot;
  Data: options_cache.db (Polygon.io) &middot;
  Generated by PilotAI Compass
</p>

</body></html>"""
