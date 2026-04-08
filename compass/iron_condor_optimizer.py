"""
Iron Condor Strategy Optimizer — Scale & Optimize across tickers, sizing,
spread widths, DTE ranges, and regime filters.

Uses ONLY real IronVault data. Reuses the proven IC backtest logic from
compass/new_strategy_explorer.py, parameterised for sweeping.

Usage::

    from compass.iron_condor_optimizer import IronCondorOptimizer
    opt = IronCondorOptimizer()
    results = opt.run()
    opt.generate_report(results, "reports/xlf_iron_condor_optimization.html")
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault

logger = logging.getLogger(__name__)

# ── Configuration space ──────────────────────────────────────────────────

# Tickers available in IronVault with enough option data for ICs
IC_TICKERS = ["SPY", "XLF", "XLI", "QQQ", "GLD", "TLT", "XLE", "XLK", "SOXX"]

# Sizing: % of $100K portfolio risked per trade
SIZING_PCTS = [0.015, 0.05, 0.10, 0.20]  # 1.5% (original), 5%, 10%, 20%

# Spread widths ($)
SPREAD_WIDTHS = [1, 2, 3, 5]

# DTE ranges: (target_dte, min_entry_offset_before_exp)
DTE_RANGES = [
    (21, 14),   # Short: ~3 weeks
    (35, 28),   # Medium: ~5 weeks (original ~37)
    (45, 35),   # Long: ~6 weeks
]

# OTM offsets: (put_otm_pct, call_otm_pct)
OTM_CONFIGS = [
    (0.05, 0.03),  # Tight
    (0.07, 0.05),  # Original
    (0.10, 0.07),  # Wide
]

# Regime filters
REGIME_FILTERS = [
    "none",        # No filtering (original)
    "low_vol",     # VIX < 20 only
    "high_vol",    # VIX > 20 only (premium richer)
    "moderate",    # 15 < VIX < 30
]

VIX_FILTER_RANGES = {
    "none": (0, 100),
    "low_vol": (0, 20),
    "high_vol": (20, 100),
    "moderate": (15, 30),
}

CAPITAL = 100_000
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ICConfig:
    """One iron condor configuration to test."""
    ticker: str
    sizing_pct: float
    spread_width: float
    target_dte: int
    min_entry_offset: int
    put_otm_pct: float
    call_otm_pct: float
    regime_filter: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = (
                f"{self.ticker}_sz{self.sizing_pct:.0%}_w{self.spread_width}"
                f"_dte{self.target_dte}_{self.regime_filter}"
            )


@dataclass
class ICYearResult:
    year: int
    n_trades: int
    total_pnl: float
    win_rate: float
    max_dd: float
    sharpe: float
    return_pct: float


@dataclass
class ICResult:
    """Result of one IC backtest configuration."""
    config: ICConfig
    trades: List[Dict]
    n_trades: int
    total_pnl: float
    win_rate: float
    max_dd: float
    sharpe: float
    cagr: float
    oos_sharpe: float
    yearly: Dict[int, ICYearResult]
    # Walk-forward: IS=2020-2022, OOS=2023-2025
    is_sharpe: float
    wf_ratio: float  # OOS/IS sharpe


@dataclass
class OptimizationResult:
    """Full optimization result across all configs."""
    configs_tested: int
    results: List[ICResult]
    best_by_sharpe: Optional[ICResult]
    best_by_cagr: Optional[ICResult]
    best_by_calmar: Optional[ICResult]
    ticker_summary: Dict[str, Dict]
    sizing_summary: Dict[str, Dict]


# ── Core backtest engine ─────────────────────────────────────────────────


def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _find_expirations(
    hd: IronVault, ticker: str, start: str, end: str, monthly_only: bool = True
) -> List[str]:
    """Find available expirations in the DB."""
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration
    """, (ticker, start, end))
    all_exps = [r[0] for r in cur.fetchall()]
    conn.close()

    if not monthly_only:
        return all_exps

    monthly = []
    last_month = ""
    for exp in all_exps:
        ym = exp[:7]
        day = int(exp[8:10])
        if ym != last_month and 15 <= day <= 21:
            monthly.append(exp)
            last_month = ym
    return monthly


def _get_underlying_prices(ticker: str) -> pd.DataFrame:
    """Fetch daily prices via Yahoo Finance (curl-safe)."""
    from backtest.backtester import _yf_download_safe
    df = _yf_download_safe(ticker, "2019-12-01", "2026-01-01")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df


def _get_vix() -> pd.Series:
    """Fetch VIX daily close."""
    from backtest.backtester import _yf_download_safe
    vix = _yf_download_safe("^VIX", "2019-12-01", "2026-01-01")
    if vix.empty:
        return pd.Series(dtype=float)
    vix.index = pd.to_datetime(vix.index)
    return vix["Close"]


def backtest_iron_condor(
    hd: IronVault,
    config: ICConfig,
    price_df: pd.DataFrame,
    vix: pd.Series,
) -> List[Dict]:
    """Run a single iron condor backtest with given config.

    All option prices from IronVault — zero synthetic data.
    """
    ticker = config.ticker
    close = price_df["Close"]
    exps = _find_expirations(hd, ticker, START_DATE, END_DATE)
    trades: List[Dict] = []
    last_entry = None

    vix_lo, vix_hi = VIX_FILTER_RANGES.get(config.regime_filter, (0, 100))

    for exp in exps:
        exp_dt_obj = _exp_dt(exp)
        entry_dt = exp_dt_obj - timedelta(days=config.target_dte)

        # Find a valid trading day near entry
        for offset in range(7):
            cand = entry_dt + timedelta(days=offset)
            cand_str = cand.strftime("%Y-%m-%d")
            if cand_str in price_df.index.strftime("%Y-%m-%d").values:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")

        # Minimum spacing between trades
        if last_entry and (entry_dt - last_entry).days < 20:
            continue

        # DTE check
        dte = (exp_dt_obj - entry_dt).days
        if dte < config.min_entry_offset:
            continue

        # VIX regime filter
        try:
            v = float(vix.loc[entry_str])
        except (KeyError, TypeError):
            v = 20.0
        if v < vix_lo or v > vix_hi:
            continue

        try:
            price = float(close.loc[entry_str])
        except (KeyError, TypeError):
            continue

        # ── Find put spread ──
        put_strikes = hd.get_available_strikes(ticker, exp, entry_str, "P")
        call_strikes = hd.get_available_strikes(ticker, exp, entry_str, "C")
        if not put_strikes or not call_strikes:
            continue

        w = config.spread_width
        put_target = price * (1 - config.put_otm_pct)
        put_short = put_long = put_credit = None

        for sk in sorted(put_strikes, key=lambda k: abs(k - put_target)):
            lk = sk - w
            if lk not in put_strikes:
                # Try nearest available
                candidates = [s for s in put_strikes if abs(s - lk) <= 0.5]
                if candidates:
                    lk = min(candidates, key=lambda s: abs(s - (sk - w)))
                else:
                    continue
            pp = hd.get_spread_prices(ticker, exp_dt_obj, sk, lk, "P", entry_str)
            if pp and pp["short_close"] - pp["long_close"] > 0.03:
                put_short = sk
                put_long = lk
                put_credit = pp["short_close"] - pp["long_close"]
                break

        if put_short is None:
            continue

        # ── Find call spread ──
        call_target = price * (1 + config.call_otm_pct)
        call_short = call_long = call_credit = None

        for sk in sorted(call_strikes, key=lambda k: abs(k - call_target)):
            lk = sk + w
            if lk not in call_strikes:
                candidates = [s for s in call_strikes if abs(s - lk) <= 0.5]
                if candidates:
                    lk = min(candidates, key=lambda s: abs(s - (sk + w)))
                else:
                    continue
            cp = hd.get_spread_prices(ticker, exp_dt_obj, sk, lk, "C", entry_str)
            if cp and cp["short_close"] - cp["long_close"] > 0.03:
                call_short = sk
                call_long = lk
                call_credit = cp["short_close"] - cp["long_close"]
                break

        if call_short is None:
            # Fallback: put-side only
            total_credit = put_credit
            max_loss = w - total_credit
        else:
            total_credit = put_credit + call_credit
            max_loss = w - total_credit  # Worst case one wing max loss

        if max_loss <= 0:
            continue

        # Position sizing
        risk_budget = CAPITAL * config.sizing_pct
        contracts = max(1, int(risk_budget / (max_loss * 100)))
        contracts = min(contracts, 50)  # Safety cap

        has_calls = call_short is not None

        # ── Walk forward to exit ──
        exit_date = exit_reason = None
        exit_total = total_credit
        hold_days = 0

        current = entry_dt + timedelta(days=1)
        while current <= exp_dt_obj:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in price_df.index.strftime("%Y-%m-%d").values:
                current += timedelta(days=1)
                continue

            hold_days += 1
            dte_rem = (exp_dt_obj - current).days

            pp = hd.get_spread_prices(
                ticker, exp_dt_obj, put_short, put_long, "P", curr_str
            )
            if pp is None:
                current += timedelta(days=1)
                continue
            cur_put_val = pp["short_close"] - pp["long_close"]

            cur_call_val = 0.0
            if has_calls:
                cp = hd.get_spread_prices(
                    ticker, exp_dt_obj, call_short, call_long, "C", curr_str
                )
                if cp is not None:
                    cur_call_val = cp["short_close"] - cp["long_close"]

            cur_total = cur_put_val + cur_call_val

            # 50% profit target
            if cur_total <= total_credit * 0.50:
                exit_date = curr_str
                exit_reason = "profit_target"
                exit_total = cur_total
                break

            # 2x stop loss
            if cur_total - total_credit > total_credit * 2.0:
                exit_date = curr_str
                exit_reason = "stop_loss"
                exit_total = cur_total
                break

            # DTE exit at 7 days
            if dte_rem <= 7:
                exit_date = curr_str
                exit_reason = "dte_exit"
                exit_total = cur_total
                break

            current += timedelta(days=1)

        if exit_date is None:
            exit_date = exp
            exit_reason = "expiration"
            exit_total = 0

        pnl = (total_credit - exit_total) * 100 * contracts

        trades.append({
            "entry_date": entry_str,
            "exit_date": exit_date,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "entry_credit": round(total_credit, 4),
            "contracts": contracts,
            "hold_days": hold_days,
        })
        last_entry = entry_dt

    return trades


def _compute_ic_result(
    config: ICConfig, trades: List[Dict], capital: float = CAPITAL
) -> ICResult:
    """Compute metrics from trade list."""
    if not trades:
        return ICResult(
            config=config, trades=[], n_trades=0, total_pnl=0, win_rate=0,
            max_dd=0, sharpe=0, cagr=0, oos_sharpe=0, yearly={},
            is_sharpe=0, wf_ratio=0,
        )

    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    n = len(pnls)
    total = pnls.sum()
    wins = (pnls > 0).sum()

    # Equity curve & DD
    equity = np.cumsum(pnls) + capital
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    # Sharpe
    mean_p = pnls.mean()
    std_p = pnls.std(ddof=1) if n > 1 else 1.0
    sharpe = float(mean_p / std_p * math.sqrt(min(n, 52))) if std_p > 0 else 0

    # CAGR
    dates = pd.to_datetime(df["exit_date"])
    entry_dates = pd.to_datetime(df["entry_date"])
    years = max((dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / capital) ** (1 / years) - 1) if total > -capital else -1.0

    # Yearly
    df["year"] = dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yn = len(yp)
        if yn == 0:
            continue
        y_eq = np.cumsum(yp) + capital
        y_pk = np.maximum.accumulate(y_eq)
        y_dd = (y_pk - y_eq) / y_pk
        y_std = yp.std(ddof=1) if yn > 1 else 1.0
        yearly[int(yr)] = ICYearResult(
            year=int(yr),
            n_trades=yn,
            total_pnl=round(float(yp.sum()), 2),
            win_rate=round(float((yp > 0).sum()) / yn, 4),
            max_dd=round(float(y_dd.max()), 4),
            sharpe=round(float(yp.mean() / y_std * math.sqrt(min(yn, 52))) if y_std > 0 else 0, 3),
            return_pct=round(float(yp.sum() / capital), 4),
        )

    # Walk-forward: IS=2020-2022, OOS=2023-2025
    is_trades = df[dates.dt.year <= 2022]["pnl"].values
    oos_trades = df[dates.dt.year >= 2023]["pnl"].values

    def _sharpe(arr):
        if len(arr) < 2:
            return 0.0
        s = arr.std(ddof=1)
        return float(arr.mean() / s * math.sqrt(min(len(arr), 52))) if s > 0 else 0

    is_sharpe = _sharpe(is_trades)
    oos_sharpe = _sharpe(oos_trades)
    wf_ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0

    return ICResult(
        config=config, trades=trades, n_trades=n,
        total_pnl=round(total, 2),
        win_rate=round(float(wins / n), 4),
        max_dd=round(max_dd, 4),
        sharpe=round(sharpe, 3),
        cagr=round(cagr, 4),
        oos_sharpe=round(oos_sharpe, 3),
        yearly=yearly,
        is_sharpe=round(is_sharpe, 3),
        wf_ratio=round(wf_ratio, 3),
    )


# ── Optimizer engine ─────────────────────────────────────────────────────


class IronCondorOptimizer:
    """Sweep iron condor configurations across tickers and parameters."""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        sizing_pcts: Optional[List[float]] = None,
        spread_widths: Optional[List[float]] = None,
        dte_ranges: Optional[List[Tuple[int, int]]] = None,
        otm_configs: Optional[List[Tuple[float, float]]] = None,
        regime_filters: Optional[List[str]] = None,
    ):
        self.tickers = tickers or list(IC_TICKERS)
        self.sizing_pcts = sizing_pcts or list(SIZING_PCTS)
        self.spread_widths = spread_widths or list(SPREAD_WIDTHS)
        self.dte_ranges = dte_ranges or list(DTE_RANGES)
        self.otm_configs = otm_configs or list(OTM_CONFIGS)
        self.regime_filters = regime_filters or list(REGIME_FILTERS)

    def _build_configs(self) -> List[ICConfig]:
        """Build all config combinations.

        To keep runtime manageable, we sweep one dimension at a time against
        the baseline, rather than full cartesian product.
        """
        configs = []

        # Baseline per ticker: original settings
        baseline_sizing = 0.015
        baseline_width = 1
        baseline_dte = (35, 28)
        baseline_otm = (0.07, 0.05)
        baseline_regime = "none"

        # 1. Ticker sweep (baseline config per ticker)
        for ticker in self.tickers:
            configs.append(ICConfig(
                ticker=ticker, sizing_pct=baseline_sizing,
                spread_width=baseline_width,
                target_dte=baseline_dte[0], min_entry_offset=baseline_dte[1],
                put_otm_pct=baseline_otm[0], call_otm_pct=baseline_otm[1],
                regime_filter=baseline_regime,
            ))

        # 2. Sizing sweep (XLF only — the known-good ticker)
        for sz in self.sizing_pcts:
            if sz == baseline_sizing:
                continue
            configs.append(ICConfig(
                ticker="XLF", sizing_pct=sz,
                spread_width=baseline_width,
                target_dte=baseline_dte[0], min_entry_offset=baseline_dte[1],
                put_otm_pct=baseline_otm[0], call_otm_pct=baseline_otm[1],
                regime_filter=baseline_regime,
            ))

        # 3. Spread width sweep (XLF)
        for w in self.spread_widths:
            if w == baseline_width:
                continue
            configs.append(ICConfig(
                ticker="XLF", sizing_pct=baseline_sizing,
                spread_width=w,
                target_dte=baseline_dte[0], min_entry_offset=baseline_dte[1],
                put_otm_pct=baseline_otm[0], call_otm_pct=baseline_otm[1],
                regime_filter=baseline_regime,
            ))

        # 4. DTE sweep (XLF)
        for dte_target, dte_min in self.dte_ranges:
            if dte_target == baseline_dte[0]:
                continue
            configs.append(ICConfig(
                ticker="XLF", sizing_pct=baseline_sizing,
                spread_width=baseline_width,
                target_dte=dte_target, min_entry_offset=dte_min,
                put_otm_pct=baseline_otm[0], call_otm_pct=baseline_otm[1],
                regime_filter=baseline_regime,
            ))

        # 5. Regime filter sweep (XLF)
        for rf in self.regime_filters:
            if rf == baseline_regime:
                continue
            configs.append(ICConfig(
                ticker="XLF", sizing_pct=baseline_sizing,
                spread_width=baseline_width,
                target_dte=baseline_dte[0], min_entry_offset=baseline_dte[1],
                put_otm_pct=baseline_otm[0], call_otm_pct=baseline_otm[1],
                regime_filter=rf,
            ))

        # 6. OTM sweep (XLF)
        for p_otm, c_otm in self.otm_configs:
            if (p_otm, c_otm) == baseline_otm:
                continue
            configs.append(ICConfig(
                ticker="XLF", sizing_pct=baseline_sizing,
                spread_width=baseline_width,
                target_dte=baseline_dte[0], min_entry_offset=baseline_dte[1],
                put_otm_pct=p_otm, call_otm_pct=c_otm,
                regime_filter=baseline_regime,
            ))

        # 7. Best combos: high sizing + best regime on top tickers
        for ticker in ["XLF", "SPY", "XLI"]:
            for sz in [0.10, 0.20]:
                for rf in ["moderate", "high_vol"]:
                    configs.append(ICConfig(
                        ticker=ticker, sizing_pct=sz,
                        spread_width=2,
                        target_dte=35, min_entry_offset=28,
                        put_otm_pct=0.07, call_otm_pct=0.05,
                        regime_filter=rf,
                    ))

        return configs

    def run(self) -> OptimizationResult:
        """Run the full optimization sweep."""
        hd = IronVault.instance()
        configs = self._build_configs()
        logger.info("Running %d IC configurations...", len(configs))

        # Cache prices per ticker
        price_cache: Dict[str, pd.DataFrame] = {}
        vix = _get_vix()

        results: List[ICResult] = []
        for i, cfg in enumerate(configs):
            ticker = cfg.ticker
            if ticker not in price_cache:
                logger.info("  Fetching %s prices...", ticker)
                price_cache[ticker] = _get_underlying_prices(ticker)
                if price_cache[ticker].empty:
                    logger.warning("  No price data for %s — skipping", ticker)

            if price_cache[ticker].empty:
                continue

            logger.info(
                "  [%d/%d] %s sz=%.0f%% w=$%d dte=%d %s",
                i + 1, len(configs), ticker,
                cfg.sizing_pct * 100, cfg.spread_width,
                cfg.target_dte, cfg.regime_filter,
            )

            trades = backtest_iron_condor(hd, cfg, price_cache[ticker], vix)
            result = _compute_ic_result(cfg, trades)
            results.append(result)

            logger.info(
                "    → %d trades, PnL=$%.0f, Sharpe=%.2f, OOS=%.2f, DD=%.1f%%",
                result.n_trades, result.total_pnl, result.sharpe,
                result.oos_sharpe, result.max_dd * 100,
            )

        # Rank results
        valid = [r for r in results if r.n_trades >= 5]
        best_sharpe = max(valid, key=lambda r: r.oos_sharpe) if valid else None
        best_cagr = max(valid, key=lambda r: r.cagr) if valid else None
        best_calmar = max(
            valid,
            key=lambda r: r.cagr / r.max_dd if r.max_dd > 0.001 else 0,
        ) if valid else None

        # Summaries
        ticker_summary = {}
        for r in results:
            t = r.config.ticker
            if t not in ticker_summary:
                ticker_summary[t] = {"configs": 0, "best_sharpe": 0, "best_cagr": 0, "trades": 0}
            ticker_summary[t]["configs"] += 1
            ticker_summary[t]["trades"] += r.n_trades
            ticker_summary[t]["best_sharpe"] = max(ticker_summary[t]["best_sharpe"], r.oos_sharpe)
            ticker_summary[t]["best_cagr"] = max(ticker_summary[t]["best_cagr"], r.cagr)

        sizing_summary = {}
        for r in results:
            if r.config.ticker != "XLF":
                continue
            sz = f"{r.config.sizing_pct:.0%}"
            if sz not in sizing_summary:
                sizing_summary[sz] = {"configs": 0, "best_sharpe": 0, "best_pnl": 0}
            sizing_summary[sz]["configs"] += 1
            sizing_summary[sz]["best_sharpe"] = max(sizing_summary[sz]["best_sharpe"], r.oos_sharpe)
            sizing_summary[sz]["best_pnl"] = max(sizing_summary[sz]["best_pnl"], r.total_pnl)

        return OptimizationResult(
            configs_tested=len(configs),
            results=results,
            best_by_sharpe=best_sharpe,
            best_by_cagr=best_cagr,
            best_by_calmar=best_calmar,
            ticker_summary=ticker_summary,
            sizing_summary=sizing_summary,
        )

    def generate_report(
        self, result: OptimizationResult, output_path: str | Path
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def save_summary(
        self, result: OptimizationResult, output_path: str | Path
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "configs_tested": result.configs_tested,
            "total_results": len(result.results),
            "best_by_oos_sharpe": _result_to_dict(result.best_by_sharpe) if result.best_by_sharpe else None,
            "best_by_cagr": _result_to_dict(result.best_by_cagr) if result.best_by_cagr else None,
            "best_by_calmar": _result_to_dict(result.best_by_calmar) if result.best_by_calmar else None,
            "ticker_summary": result.ticker_summary,
            "sizing_summary": result.sizing_summary,
            "all_results": [_result_to_dict(r) for r in result.results],
        }
        output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return output_path


def _result_to_dict(r: ICResult) -> Dict:
    return {
        "ticker": r.config.ticker,
        "sizing_pct": r.config.sizing_pct,
        "spread_width": r.config.spread_width,
        "target_dte": r.config.target_dte,
        "put_otm_pct": r.config.put_otm_pct,
        "call_otm_pct": r.config.call_otm_pct,
        "regime_filter": r.config.regime_filter,
        "n_trades": r.n_trades,
        "total_pnl": r.total_pnl,
        "win_rate": r.win_rate,
        "max_dd": r.max_dd,
        "sharpe": r.sharpe,
        "cagr": r.cagr,
        "oos_sharpe": r.oos_sharpe,
        "is_sharpe": r.is_sharpe,
        "wf_ratio": r.wf_ratio,
    }


# ── HTML Report ──────────────────────────────────────────────────────────


def _fc(v: float) -> str:
    color = "#22c55e" if v > 0 else "#ef4444"
    return f'<span style="color:{color}">{v:+.1f}%</span>'


def _fp(v: float) -> str:
    return f"{v:.1f}%"


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _build_html(result: OptimizationResult) -> str:
    bs = result.best_by_sharpe
    bc = result.best_by_cagr
    bk = result.best_by_calmar

    # All results table (sorted by OOS Sharpe)
    sorted_results = sorted(result.results, key=lambda r: r.oos_sharpe, reverse=True)
    all_rows = ""
    for r in sorted_results:
        c = r.config
        calmar = r.cagr / r.max_dd if r.max_dd > 0.001 else 0
        wf_color = "#22c55e" if r.wf_ratio > 0.5 else ("#f59e0b" if r.wf_ratio > 0 else "#ef4444")
        all_rows += (
            f"<tr><td style='text-align:left'>{c.ticker}</td>"
            f"<td>{c.sizing_pct:.0%}</td>"
            f"<td>${c.spread_width}</td>"
            f"<td>{c.target_dte}d</td>"
            f"<td>{c.put_otm_pct:.0%}/{c.call_otm_pct:.0%}</td>"
            f"<td>{c.regime_filter}</td>"
            f"<td>{r.n_trades}</td>"
            f"<td style='color:{'#22c55e' if r.total_pnl > 0 else '#ef4444'}'>${r.total_pnl:,.0f}</td>"
            f"<td>{r.win_rate:.0%}</td>"
            f"<td style='color:#f59e0b'>{r.max_dd:.1%}</td>"
            f"<td>{_fr(r.sharpe)}</td>"
            f"<td><strong>{_fr(r.oos_sharpe)}</strong></td>"
            f"<td style='color:{wf_color}'>{_fr(r.wf_ratio)}</td>"
            f"<td>{r.cagr:.1%}</td>"
            f"<td>{calmar:.1f}</td></tr>\n"
        )

    # Ticker comparison
    ticker_rows = ""
    for t, s in sorted(result.ticker_summary.items(), key=lambda x: x[1]["best_sharpe"], reverse=True):
        ticker_rows += (
            f"<tr><td style='text-align:left'>{t}</td>"
            f"<td>{s['configs']}</td>"
            f"<td>{s['trades']}</td>"
            f"<td><strong>{_fr(s['best_sharpe'])}</strong></td>"
            f"<td>{s['best_cagr']:.1%}</td></tr>\n"
        )

    # Sizing impact
    sizing_rows = ""
    for sz, s in sorted(result.sizing_summary.items()):
        sizing_rows += (
            f"<tr><td>{sz}</td>"
            f"<td>{s['configs']}</td>"
            f"<td>{_fr(s['best_sharpe'])}</td>"
            f"<td>${s['best_pnl']:,.0f}</td></tr>\n"
        )

    # Best config details
    def _best_card(label, r):
        if r is None:
            return "<p>No valid result</p>"
        c = r.config
        return (
            f"<div class='c'><div class='l'>{label}</div>"
            f"<div class='v'>{c.ticker} sz={c.sizing_pct:.0%} w=${c.spread_width} "
            f"dte={c.target_dte} {c.regime_filter}</div>"
            f"<div style='color:#8b949e;font-size:.8em'>"
            f"PnL=${r.total_pnl:,.0f} | Sharpe={_fr(r.sharpe)} | "
            f"OOS={_fr(r.oos_sharpe)} | WR={r.win_rate:.0%} | "
            f"DD={r.max_dd:.1%} | CAGR={r.cagr:.1%}</div></div>"
        )

    # Year-by-year for best config
    best_yearly_rows = ""
    if bs and bs.yearly:
        for yr in sorted(bs.yearly.keys()):
            y = bs.yearly[yr]
            best_yearly_rows += (
                f"<tr><td>{yr}</td><td>{y.n_trades}</td>"
                f"<td style='color:{'#22c55e' if y.total_pnl > 0 else '#ef4444'}'>${y.total_pnl:,.0f}</td>"
                f"<td>{y.win_rate:.0%}</td>"
                f"<td style='color:#f59e0b'>{y.max_dd:.1%}</td>"
                f"<td>{_fr(y.sharpe)}</td>"
                f"<td>{y.return_pct:.2%}</td></tr>\n"
            )

    hero_color = "#3fb950" if bs and bs.oos_sharpe > 1.5 else "#d29922"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Iron Condor Optimization Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2,h3{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid {hero_color};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:1.8em;font-weight:800;color:{hero_color}}}
.hero .sub{{color:#8b949e;margin-top:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1em;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.85em}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22;font-size:.8em;position:sticky;top:0}}
td:first-child,th:first-child{{text-align:left}}
tr:hover td{{background:#161b2280}}
.section{{margin:32px 0}}
.note{{color:#8b949e;font-size:.85em;margin:8px 0}}
.warn{{background:#d2992220;border:1px solid #d29922;border-radius:8px;padding:12px;margin:16px 0;color:#d29922}}
</style></head><body>

<h1>Iron Condor Optimization Report</h1>
<p class="note">
  Data: <strong>IronVault</strong> (real Polygon prices) &middot;
  {result.configs_tested} configurations tested &middot;
  {len(IC_TICKERS)} tickers &middot;
  2020–2025
</p>

<div class="hero">
  <div class="big">Best OOS Sharpe: {_fr(bs.oos_sharpe) if bs else 'N/A'}</div>
  <div class="sub">
    {f'{bs.config.ticker} | sz={bs.config.sizing_pct:.0%} | w=${bs.config.spread_width} | dte={bs.config.target_dte} | {bs.config.regime_filter}' if bs else 'No valid result'}
    {f' | PnL=${bs.total_pnl:,.0f} | WR={bs.win_rate:.0%} | DD={bs.max_dd:.1%}' if bs else ''}
  </div>
</div>

<div class="cards">
  {_best_card("Best OOS Sharpe", bs)}
  {_best_card("Best CAGR", bc)}
  {_best_card("Best Calmar", bk)}
</div>

<div class="section">
<h2>Ticker Comparison</h2>
<p class="note">Which underlying works best for iron condors?</p>
<table>
<thead><tr><th>Ticker</th><th>Configs</th><th>Trades</th><th>Best OOS Sharpe</th><th>Best CAGR</th></tr></thead>
<tbody>{ticker_rows}</tbody></table>
</div>

<div class="section">
<h2>Position Sizing Impact (XLF)</h2>
<p class="note">How does scaling from 1.5% to 20% per trade affect results?</p>
<table>
<thead><tr><th>Sizing</th><th>Configs</th><th>Best OOS Sharpe</th><th>Best PnL</th></tr></thead>
<tbody>{sizing_rows}</tbody></table>
</div>

<div class="section">
<h2>Best Config: Year-by-Year</h2>
<table>
<thead><tr><th>Year</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Max DD</th><th>Sharpe</th><th>Return</th></tr></thead>
<tbody>{best_yearly_rows}</tbody></table>
</div>

<div class="section">
<h2>All Configurations (sorted by OOS Sharpe)</h2>
<div style="overflow-x:auto">
<table>
<thead><tr>
<th>Ticker</th><th>Sizing</th><th>Width</th><th>DTE</th><th>OTM P/C</th><th>Regime</th>
<th>Trades</th><th>PnL</th><th>WR</th><th>DD</th><th>Sharpe</th><th>OOS Sharpe</th>
<th>WF Ratio</th><th>CAGR</th><th>Calmar</th>
</tr></thead>
<tbody>{all_rows}</tbody></table>
</div>
</div>

<div class="warn">
  <strong>Walk-Forward Validation</strong>: IS=2020-2022, OOS=2023-2025.
  WF Ratio = OOS Sharpe / IS Sharpe. Values &gt; 0.5 suggest robustness.
  Values &lt; 0 indicate overfitting.
</div>

<p class="note" style="margin-top:40px;text-align:center">
  Iron Condor Optimization &middot; All prices from IronVault (options_cache.db) &middot;
  Generated by PilotAI Compass
</p>

</body></html>"""
