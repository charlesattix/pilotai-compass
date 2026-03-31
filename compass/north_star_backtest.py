"""
North Star backtest — end-to-end validation pipeline.

Wires together the full compass module chain:
  features → signal → sizing → risk_gate → execution → attribution

Runs walk-forward expanding-window validation across 2020-2025 on
EXP-400/401/combined data.  Evaluates against North Star targets:
  - Annual return target (default 55%)
  - Sharpe ratio target  (default 6.0)
  - Max drawdown limit   (default 30%)

Generates comprehensive HTML report with equity curve, drawdown chart,
per-year table, regime overlay, and target scorecard.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.north_star_backtest import NorthStarBacktest
    bt = NorthStarBacktest()
    result = bt.run(trades_df)
    NorthStarBacktest.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "north_star_backtest.html"
TRADING_DAYS = 252


# ── North Star targets ───────────────────────────────────────────────────


@dataclass
class NorthStarTargets:
    """Configurable North Star performance targets."""

    annual_return_pct: float = 55.0
    sharpe_ratio: float = 6.0
    max_drawdown_pct: float = 30.0


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Pipeline configuration."""

    targets: NorthStarTargets = field(default_factory=NorthStarTargets)
    initial_capital: float = 100_000.0
    # Signal
    signal_threshold: float = 0.4
    # Sizing
    base_contracts: int = 5
    max_position_pct: float = 0.05
    # Risk
    max_drawdown_halt: float = 0.25
    regime_filter: bool = True
    allowed_regimes: List[str] = field(default_factory=lambda: ["bull", "sideways"])
    # Execution
    slippage_bps: float = 5.0
    commission_per_contract: float = 1.30
    # Walk-forward
    min_train_days: int = 252  # 1 year minimum training


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    """Single trade with attribution."""

    trade_id: int
    entry_date: str
    exit_date: str
    year: int
    strategy_type: str
    regime: str
    contracts: int
    gross_pnl: float
    slippage: float
    commission: float
    net_pnl: float
    signal_score: float
    win: bool


@dataclass
class YearMetrics:
    """Per-year performance breakdown."""

    year: int
    n_trades: int
    total_pnl: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    avg_pnl: float
    profit_factor: float
    # Target checks
    return_target_met: bool
    sharpe_target_met: bool
    drawdown_target_met: bool
    all_targets_met: bool


@dataclass
class RegimeMetrics:
    """Performance within a regime."""

    regime: str
    n_trades: int
    total_pnl: float
    win_rate: float
    sharpe: float


@dataclass
class WalkForwardFold:
    """Walk-forward expanding-window fold."""

    fold: int
    train_years: List[int]
    test_year: int
    train_sharpe: float
    test_sharpe: float
    train_win_rate: float
    test_win_rate: float
    test_pnl: float
    test_return_pct: float
    n_train: int
    n_test: int


@dataclass
class NorthStarResult:
    """Full backtest result."""

    config: BacktestConfig
    trades: List[TradeResult]
    year_metrics: List[YearMetrics]
    regime_metrics: List[RegimeMetrics]
    walk_forward_folds: List[WalkForwardFold]
    # Aggregates
    total_pnl: float
    total_return_pct: float
    overall_sharpe: float
    overall_sortino: float
    overall_max_dd_pct: float
    overall_win_rate: float
    overall_profit_factor: float
    n_trades: int
    n_years: int
    initial_capital: float
    final_capital: float
    equity_curve: np.ndarray
    # Target assessment
    annual_return_target_met: bool
    sharpe_target_met: bool
    drawdown_target_met: bool
    all_targets_met: bool
    years_hitting_all: int
    target_hit_rate: float


# ── Feature / signal pipeline ────────────────────────────────────────────


def compute_signal_score(row: pd.Series) -> float:
    """Simple signal scoring from available features.

    Combines regime, VIX percentile, IV rank, momentum into a 0-1 score.
    """
    score = 0.5  # baseline

    # Regime bonus
    regime = str(row.get("regime", "unknown")).lower()
    if regime == "bull":
        score += 0.15
    elif regime == "sideways":
        score += 0.05
    elif regime == "bear":
        score -= 0.10

    # VIX percentile (higher = more premium, good for selling)
    vix_pct = float(row.get("vix_percentile_50d", 50))
    if vix_pct > 70:
        score += 0.10
    elif vix_pct < 30:
        score -= 0.05

    # IV rank (higher = richer premium)
    iv_rank = float(row.get("iv_rank", 50))
    if iv_rank > 50:
        score += 0.08
    elif iv_rank < 20:
        score -= 0.05

    # Momentum (positive = trending up, good for CS)
    mom = float(row.get("momentum_5d_pct", 0))
    if mom > 0:
        score += 0.05
    elif mom < -2:
        score -= 0.05

    # Historical win indicator
    if "win" in row and not pd.isna(row["win"]):
        score += 0.05 * float(row["win"])

    return max(0.0, min(1.0, score))


# ── Sizing ───────────────────────────────────────────────────────────────


def compute_contracts(
    capital: float,
    signal: float,
    entry_price: float,
    config: BacktestConfig,
) -> int:
    """Compute position size in contracts."""
    if entry_price <= 0 or capital <= 0:
        return config.base_contracts

    frac = config.max_position_pct * signal
    notional = capital * frac
    per_contract = abs(entry_price) * 100
    if per_contract <= 0:
        return config.base_contracts
    return max(1, int(notional / per_contract))


# ── Risk gate ────────────────────────────────────────────────────────────


def risk_gate(
    regime: str,
    drawdown: float,
    config: BacktestConfig,
) -> Tuple[bool, str]:
    """Check if trade passes risk gates."""
    if drawdown >= config.max_drawdown_halt:
        return False, f"drawdown_halt({drawdown:.1%})"
    if config.regime_filter and regime not in config.allowed_regimes:
        return False, f"regime_blocked({regime})"
    return True, ""


# ── Execution ────────────────────────────────────────────────────────────


def apply_costs(
    gross_pnl: float,
    contracts: int,
    entry_price: float,
    exit_price: float,
    config: BacktestConfig,
) -> Tuple[float, float, float]:
    """Apply slippage and commissions.  Returns (slippage, commission, net_pnl)."""
    multiplier = contracts * 100
    slip = (abs(entry_price) + abs(exit_price)) * config.slippage_bps / 10_000 * multiplier
    comm = config.commission_per_contract * contracts * 2
    net = gross_pnl - slip - comm
    return slip, comm, net


# ── Metrics ──────────────────────────────────────────────────────────────


def compute_sharpe(pnl_array: np.ndarray) -> float:
    if len(pnl_array) < 2:
        return 0.0
    mu = pnl_array.mean()
    std = pnl_array.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mu / std * math.sqrt(TRADING_DAYS))


def compute_sortino(pnl_array: np.ndarray) -> float:
    if len(pnl_array) < 2:
        return 0.0
    mu = pnl_array.mean()
    down = pnl_array[pnl_array < 0]
    if len(down) == 0:
        return float("inf") if mu > 0 else 0.0
    ds = np.sqrt(np.mean(down ** 2))
    if ds < 1e-12:
        return 0.0
    return float(mu / ds * math.sqrt(TRADING_DAYS))


def compute_max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()))


def compute_profit_factor(pnl_array: np.ndarray) -> float:
    gains = pnl_array[pnl_array > 0].sum()
    losses = abs(pnl_array[pnl_array < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ── Walk-forward validation ──────────────────────────────────────────────


def walk_forward_by_year(
    trades: List[TradeResult],
    years: List[int],
    capital: float,
) -> List[WalkForwardFold]:
    """Expanding-window walk-forward by year."""
    if len(years) < 2:
        return []

    folds: List[WalkForwardFold] = []
    sorted_years = sorted(years)

    for i in range(1, len(sorted_years)):
        train_years = sorted_years[:i]
        test_year = sorted_years[i]

        train_trades = [t for t in trades if t.year in train_years]
        test_trades = [t for t in trades if t.year == test_year]

        if not train_trades or not test_trades:
            continue

        train_pnls = np.array([t.net_pnl for t in train_trades])
        test_pnls = np.array([t.net_pnl for t in test_trades])

        test_return = float(test_pnls.sum() / capital * 100)

        folds.append(WalkForwardFold(
            fold=i,
            train_years=train_years,
            test_year=test_year,
            train_sharpe=compute_sharpe(train_pnls),
            test_sharpe=compute_sharpe(test_pnls),
            train_win_rate=sum(1 for t in train_trades if t.win) / len(train_trades),
            test_win_rate=sum(1 for t in test_trades if t.win) / len(test_trades),
            test_pnl=float(test_pnls.sum()),
            test_return_pct=test_return,
            n_train=len(train_trades),
            n_test=len(test_trades),
        ))

    return folds


# ── Core engine ──────────────────────────────────────────────────────────


class NorthStarBacktest:
    """End-to-end North Star validation pipeline."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, trades_df: pd.DataFrame) -> NorthStarResult:
        """Run full pipeline on trade data.

        Expected columns: entry_date, exit_date, year, pnl, strategy_type,
        regime, vix, iv_rank, net_credit, contracts, win, etc.
        """
        cfg = self.config

        if trades_df.empty:
            return self._empty_result()

        df = trades_df.copy()

        # Ensure required columns
        if "year" not in df.columns and "entry_date" in df.columns:
            df["year"] = pd.to_datetime(df["entry_date"]).dt.year
        if "regime" not in df.columns:
            df["regime"] = "unknown"
        df["regime"] = df["regime"].fillna("unknown")

        # Stage 1: Signal scoring
        df["signal_score"] = df.apply(compute_signal_score, axis=1)

        # Stage 2: Signal filter
        df = df[df["signal_score"] >= cfg.signal_threshold].copy()
        if df.empty:
            return self._empty_result()

        # Stage 3-6: Process trades through pipeline
        capital = cfg.initial_capital
        current_capital = capital
        peak_capital = capital
        trade_results: List[TradeResult] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            regime = str(row.get("regime", "unknown"))
            dd = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0

            # Risk gate
            passed, _ = risk_gate(regime, dd, cfg)
            if not passed:
                continue

            # Signal & sizing
            signal = float(row.get("signal_score", 0.5))
            entry_price = abs(float(row.get("net_credit", row.get("entry_price", 1.0))))
            contracts = compute_contracts(current_capital, signal, entry_price, cfg)

            # Scale PnL to our contract count
            orig_contracts = int(row.get("contracts", cfg.base_contracts))
            gross_pnl = float(row.get("pnl", 0.0))
            if orig_contracts > 0 and orig_contracts != contracts:
                gross_pnl = gross_pnl / orig_contracts * contracts

            # Execution costs
            exit_price = entry_price  # approximate
            slip, comm, net_pnl = apply_costs(
                gross_pnl, contracts, entry_price, exit_price, cfg
            )

            trade_results.append(TradeResult(
                trade_id=idx,
                entry_date=str(row.get("entry_date", "")),
                exit_date=str(row.get("exit_date", "")),
                year=int(row.get("year", 0)),
                strategy_type=str(row.get("strategy_type", "unknown")),
                regime=regime,
                contracts=contracts,
                gross_pnl=gross_pnl,
                slippage=slip,
                commission=comm,
                net_pnl=net_pnl,
                signal_score=signal,
                win=net_pnl > 0,
            ))

            current_capital += net_pnl
            peak_capital = max(peak_capital, current_capital)

        if not trade_results:
            return self._empty_result()

        # Build equity curve
        pnls = np.array([t.net_pnl for t in trade_results])
        equity = capital + np.cumsum(pnls)

        # Per-year metrics
        years = sorted(set(t.year for t in trade_results))
        year_metrics = self._compute_year_metrics(trade_results, years, capital)

        # Regime metrics
        regime_metrics = self._compute_regime_metrics(trade_results)

        # Walk-forward
        wf_folds = walk_forward_by_year(trade_results, years, capital)

        # Aggregates
        n = len(trade_results)
        wins = sum(1 for t in trade_results if t.win)
        total_pnl = float(pnls.sum())
        overall_dd = compute_max_drawdown_pct(equity)

        # Target assessment
        targets = cfg.targets
        avg_annual = np.mean([ym.annual_return_pct for ym in year_metrics]) if year_metrics else 0.0
        overall_sharpe = compute_sharpe(pnls)
        ret_met = bool(avg_annual >= targets.annual_return_pct)
        sharpe_met = bool(overall_sharpe >= targets.sharpe_ratio)
        dd_met = bool(overall_dd * 100 <= targets.max_drawdown_pct)
        years_all = sum(1 for ym in year_metrics if ym.all_targets_met)

        return NorthStarResult(
            config=cfg,
            trades=trade_results,
            year_metrics=year_metrics,
            regime_metrics=regime_metrics,
            walk_forward_folds=wf_folds,
            total_pnl=total_pnl,
            total_return_pct=total_pnl / capital * 100,
            overall_sharpe=overall_sharpe,
            overall_sortino=compute_sortino(pnls),
            overall_max_dd_pct=overall_dd * 100,
            overall_win_rate=wins / n if n > 0 else 0.0,
            overall_profit_factor=compute_profit_factor(pnls),
            n_trades=n,
            n_years=len(years),
            initial_capital=capital,
            final_capital=float(equity[-1]),
            equity_curve=equity,
            annual_return_target_met=ret_met,
            sharpe_target_met=sharpe_met,
            drawdown_target_met=dd_met,
            all_targets_met=ret_met and sharpe_met and dd_met,
            years_hitting_all=years_all,
            target_hit_rate=years_all / len(year_metrics) if year_metrics else 0.0,
        )

    def _compute_year_metrics(
        self, trades: List[TradeResult], years: List[int], capital: float,
    ) -> List[YearMetrics]:
        targets = self.config.targets
        results: List[YearMetrics] = []

        for year in years:
            yt = [t for t in trades if t.year == year]
            if not yt:
                continue
            pnls = np.array([t.net_pnl for t in yt])
            equity = capital + np.cumsum(pnls)
            total = float(pnls.sum())
            ann_ret = total / capital * 100
            sharpe = compute_sharpe(pnls)
            dd = compute_max_drawdown_pct(equity) * 100
            n = len(yt)
            wins = sum(1 for t in yt if t.win)
            pf = compute_profit_factor(pnls)

            ret_met = bool(ann_ret >= targets.annual_return_pct)
            sh_met = bool(sharpe >= targets.sharpe_ratio)
            dd_met = bool(dd <= targets.max_drawdown_pct)

            results.append(YearMetrics(
                year=year, n_trades=n, total_pnl=total,
                annual_return_pct=ann_ret, sharpe=sharpe,
                max_drawdown_pct=dd, win_rate=wins / n,
                avg_pnl=total / n, profit_factor=pf,
                return_target_met=ret_met, sharpe_target_met=sh_met,
                drawdown_target_met=dd_met,
                all_targets_met=ret_met and sh_met and dd_met,
            ))

        return results

    def _compute_regime_metrics(
        self, trades: List[TradeResult],
    ) -> List[RegimeMetrics]:
        by_regime: Dict[str, List[TradeResult]] = {}
        for t in trades:
            by_regime.setdefault(t.regime, []).append(t)

        results: List[RegimeMetrics] = []
        for regime, rt in sorted(by_regime.items()):
            pnls = np.array([t.net_pnl for t in rt])
            n = len(rt)
            results.append(RegimeMetrics(
                regime=regime, n_trades=n,
                total_pnl=float(pnls.sum()),
                win_rate=sum(1 for t in rt if t.win) / n,
                sharpe=compute_sharpe(pnls),
            ))
        return results

    def _empty_result(self) -> NorthStarResult:
        cfg = self.config
        return NorthStarResult(
            config=cfg, trades=[], year_metrics=[], regime_metrics=[],
            walk_forward_folds=[], total_pnl=0.0, total_return_pct=0.0,
            overall_sharpe=0.0, overall_sortino=0.0, overall_max_dd_pct=0.0,
            overall_win_rate=0.0, overall_profit_factor=0.0,
            n_trades=0, n_years=0, initial_capital=cfg.initial_capital,
            final_capital=cfg.initial_capital, equity_curve=np.array([cfg.initial_capital]),
            annual_return_target_met=False, sharpe_target_met=False,
            drawdown_target_met=False, all_targets_met=False,
            years_hitting_all=0, target_hit_rate=0.0,
        )

    @staticmethod
    def generate_report(
        result: NorthStarResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fd(v: float) -> str:
    return f"${v:,.2f}"


def _fp(v: float) -> str:
    return f"{v:.1f}%"


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _target_icon(met: bool) -> str:
    return '<span style="color:#3fb950">&#10003;</span>' if met else '<span style="color:#f85149">&#10007;</span>'


def _svg_line(values, title, color="#58a6ff", w=700, h=200):
    vals = list(values)
    if len(vals) < 2:
        return ""
    n = len(vals)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65
    y_min = min(vals)
    y_max = max(vals)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def tx(i): return pad + i / max(n - 1, 1) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(n))
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _year_table(year_metrics: List[YearMetrics], targets: NorthStarTargets) -> str:
    if not year_metrics:
        return "<p class='meta'>No yearly data.</p>"
    rows = ""
    for ym in year_metrics:
        rows += f"""<tr>
          <td>{ym.year}</td><td>{ym.n_trades}</td>
          <td>{_fp(ym.annual_return_pct)} {_target_icon(ym.return_target_met)}</td>
          <td>{_fr(ym.sharpe)} {_target_icon(ym.sharpe_target_met)}</td>
          <td>{_fp(ym.max_drawdown_pct)} {_target_icon(ym.drawdown_target_met)}</td>
          <td>{_fp(ym.win_rate * 100)}</td><td>{_fd(ym.total_pnl)}</td>
          <td>{_fr(ym.profit_factor)}</td>
          <td>{_target_icon(ym.all_targets_met)}</td>
        </tr>"""
    return f"""<table class="data-table">
      <tr><th>Year</th><th>Trades</th>
          <th>Return (tgt {_fp(targets.annual_return_pct)})</th>
          <th>Sharpe (tgt {_fr(targets.sharpe_ratio)})</th>
          <th>Max DD (tgt {_fp(targets.max_drawdown_pct)})</th>
          <th>Win Rate</th><th>Total PnL</th><th>PF</th><th>All</th></tr>
      {rows}</table>"""


def _regime_table(regimes: List[RegimeMetrics]) -> str:
    if not regimes:
        return ""
    rows = "".join(
        f"<tr><td style='text-align:left'>{r.regime}</td><td>{r.n_trades}</td><td>{_fd(r.total_pnl)}</td><td>{_fp(r.win_rate * 100)}</td><td>{_fr(r.sharpe)}</td></tr>"
        for r in regimes
    )
    return f"""<table class="data-table"><tr><th style='text-align:left'>Regime</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th></tr>{rows}</table>"""


def _wf_table(folds: List[WalkForwardFold]) -> str:
    if not folds:
        return "<p class='meta'>Insufficient data for walk-forward.</p>"
    rows = ""
    for f in folds:
        decay_color = "#3fb950" if f.test_sharpe > 0 else "#f85149"
        rows += f"""<tr><td>{f.fold}</td><td>{','.join(str(y) for y in f.train_years)}</td>
          <td>{f.test_year}</td><td>{f.n_train}</td><td>{f.n_test}</td>
          <td>{_fr(f.train_sharpe)}</td><td style="color:{decay_color}">{_fr(f.test_sharpe)}</td>
          <td>{_fp(f.test_return_pct)}</td></tr>"""
    return f"""<table class="data-table"><tr><th>Fold</th><th>Train</th><th>Test</th>
          <th>N Train</th><th>N Test</th><th>Train Sharpe</th><th>Test Sharpe</th><th>Test Return</th></tr>{rows}</table>"""


def _scorecard(result: NorthStarResult) -> str:
    t = result.config.targets
    return f"""
    <div class="scorecard">
      <div class="score-row">
        <span class="label">Annual Return</span>
        <span class="value">{_fp(result.total_return_pct / max(result.n_years, 1))}</span>
        <span class="target">Target: {_fp(t.annual_return_pct)}</span>
        <span>{_target_icon(result.annual_return_target_met)}</span>
      </div>
      <div class="score-row">
        <span class="label">Sharpe Ratio</span>
        <span class="value">{_fr(result.overall_sharpe)}</span>
        <span class="target">Target: {_fr(t.sharpe_ratio)}</span>
        <span>{_target_icon(result.sharpe_target_met)}</span>
      </div>
      <div class="score-row">
        <span class="label">Max Drawdown</span>
        <span class="value">{_fp(result.overall_max_dd_pct)}</span>
        <span class="target">Target: &le;{_fp(t.max_drawdown_pct)}</span>
        <span>{_target_icon(result.drawdown_target_met)}</span>
      </div>
      <div class="score-row">
        <span class="label">Years Hitting All</span>
        <span class="value">{result.years_hitting_all}/{result.n_years}</span>
        <span class="target">Hit Rate: {_fp(result.target_hit_rate * 100)}</span>
        <span>{_target_icon(result.all_targets_met)}</span>
      </div>
    </div>"""


def _build_html(result: NorthStarResult) -> str:
    cfg = result.config
    overall_color = "#3fb950" if result.all_targets_met else "#f85149"

    # Equity curve
    equity_list = result.equity_curve.tolist()
    # Drawdown curve
    peak = np.maximum.accumulate(result.equity_curve)
    dd_curve = ((result.equity_curve - peak) / np.where(peak > 0, peak, 1) * 100).tolist()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>North Star Backtest</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .hero {{ background: #161b22; border: 2px solid {overall_color}; border-radius: 12px;
           padding: 24px; text-align: center; margin: 20px 0; }}
  .hero .big {{ font-size: 2.5em; font-weight: 800; color: {overall_color}; }}
  .hero .sub {{ color: #8b949e; font-size: 1.1em; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
              gap: 10px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 10px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  .scorecard {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 20px; margin: 16px 0; }}
  .score-row {{ display: flex; align-items: center; gap: 16px; padding: 8px 0;
                border-bottom: 1px solid #21262d; }}
  .score-row .label {{ width: 140px; color: #8b949e; }}
  .score-row .value {{ width: 80px; font-weight: 700; color: #f0f6fc; }}
  .score-row .target {{ flex: 1; color: #8b949e; font-size: 0.9em; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
</style>
</head>
<body>
<h1>North Star Backtest</h1>

<div class="hero">
  <div class="big">{"ALL TARGETS MET" if result.all_targets_met else "TARGETS NOT MET"}</div>
  <div class="sub">{result.n_trades} trades &middot; {result.n_years} years &middot;
     {_fd(result.initial_capital)} &rarr; {_fd(result.final_capital)}</div>
</div>

<div class="summary">
  <div class="stat"><div class="label">Total Return</div><div class="value">{_fp(result.total_return_pct)}</div></div>
  <div class="stat"><div class="label">Sharpe</div><div class="value">{_fr(result.overall_sharpe)}</div></div>
  <div class="stat"><div class="label">Sortino</div><div class="value">{_fr(result.overall_sortino)}</div></div>
  <div class="stat"><div class="label">Max DD</div><div class="value">{_fp(result.overall_max_dd_pct)}</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">{_fp(result.overall_win_rate * 100)}</div></div>
  <div class="stat"><div class="label">Profit Factor</div><div class="value">{_fr(result.overall_profit_factor)}</div></div>
  <div class="stat"><div class="label">Total PnL</div><div class="value">{_fd(result.total_pnl)}</div></div>
</div>

<h2>North Star Scorecard</h2>
{_scorecard(result)}

<h2>Equity Curve</h2>
{_svg_line(equity_list, "Equity Curve ($)", "#3fb950")}

<h2>Drawdown</h2>
{_svg_line(dd_curve, "Drawdown (%)", "#f85149")}

<h2>Per-Year Performance</h2>
{_year_table(result.year_metrics, cfg.targets)}

<h2>Regime Performance</h2>
{_regime_table(result.regime_metrics)}

<h2>Walk-Forward Validation</h2>
{_wf_table(result.walk_forward_folds)}

</body>
</html>"""
