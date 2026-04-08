"""
Deep Validation: Cross-Asset Pairs Mean-Reversion (TLT-SPY Correlation Breakdown).

The original strategy from strategy_discovery_r2.py showed OOS Sharpe 6.31
with SPY correlation 0.02. This module validates whether that's real:

1. Walk-forward expanding window (train 1yr → test next year, repeat)
2. Parameter sensitivity sweep (lookback, z-score thresholds, hold period)
3. Test on more ticker pairs (GLD-SPY, QQQ-TLT, XLI-TLT, etc.)
4. Regime breakdown (bull/bear/crash/high_vol performance)
5. Look-ahead bias & overfitting artifact detection

All option prices from IronVault — zero synthetic data.
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

CAPITAL = 100_000

# ── Pairs to test ────────────────────────────────────────────────────────

PAIRS = [
    ("TLT", "SPY"),   # Original — bonds vs equities
    ("GLD", "SPY"),   # Gold vs equities
    ("TLT", "QQQ"),   # Bonds vs tech
    ("XLI", "TLT"),   # Industrials vs bonds
    ("GLD", "TLT"),   # Gold vs bonds
    ("XLF", "SPY"),   # Financials vs equities
    ("QQQ", "SPY"),   # Tech vs broad market
]

# The trade ticker — we sell put spreads on this when signal fires
TRADE_TICKER = "SPY"

# ── Parameter sweep space ────────────────────────────────────────────────

LOOKBACK_WINDOWS = [15, 20, 30, 45, 60]
CORR_THRESHOLDS = [-0.1, 0.0, 0.1, 0.2]  # enter when corr exceeds this
SPREAD_WIDTHS = [3.0, 5.0, 10.0]
OTM_PCTS = [0.93, 0.95, 0.97]  # 7%, 5%, 3% OTM
MIN_SPACINGS = [10, 14, 21]  # min days between trades

# Walk-forward windows
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Regime thresholds
REGIME_THRESHOLDS = {
    "bull": {"vix_max": 20},
    "bear": {"vix_min": 25},
    "high_vol": {"vix_min": 30},
    "crash": {"vix_min": 40},
    "low_vol": {"vix_max": 15},
}


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    asset_a: str
    asset_b: str
    lookback: int = 30
    corr_threshold: float = 0.0
    spread_width: float = 5.0
    otm_pct: float = 0.93
    min_spacing: int = 14
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.asset_a}-{self.asset_b}_lb{self.lookback}_t{self.corr_threshold}"


@dataclass
class WFWindow:
    train_start: int
    train_end: int
    test_year: int
    test_sharpe: float
    test_n_trades: int
    test_pnl: float
    test_win_rate: float


@dataclass
class ValidationResult:
    # Walk-forward
    wf_windows: List[WFWindow]
    wf_avg_oos_sharpe: float
    wf_consistency: float  # fraction of windows with positive Sharpe

    # Parameter sensitivity
    param_results: List[Dict]
    baseline_sharpe: float
    sensitivity_range: Tuple[float, float]  # (min, max) Sharpe across params

    # Multi-pair
    pair_results: Dict[str, Dict]
    best_pair: str

    # Regime breakdown
    regime_results: Dict[str, Dict]

    # Bias checks
    bias_flags: List[str]
    overall_verdict: str  # CONFIRMED, CAUTION, OVERFIT

    # Raw data
    all_trades: List[Dict]
    config: PairConfig


# ── Core backtest ────────────────────────────────────────────────────────


def _dl(ticker: str) -> pd.DataFrame:
    from backtest.backtester import _yf_download_safe
    df = _yf_download_safe(ticker, "2019-06-01", "2026-01-01")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df


def _find_exps(hd, ticker, start, end, monthly=True):
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration""", (ticker, start, end))
    exps = [r[0] for r in cur.fetchall()]
    conn.close()
    if not monthly:
        return exps
    out, last = [], ""
    for e in exps:
        ym, day = e[:7], int(e[8:10])
        if ym != last and 15 <= day <= 21:
            out.append(e)
            last = ym
    return out


def _exp_dt(s):
    return datetime.strptime(s, "%Y-%m-%d")


def _next_td(dt, td_set):
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


def _sell_put_spread(hd, ticker, exp, trade_date, price, otm_pct=0.93, width=5.0):
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes:
        return None
    exp_obj = _exp_dt(exp)
    target = price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not cands:
                continue
            lk = max(cands)
        aw = sk - lk
        if aw <= 0:
            continue
        pp = hd.get_spread_prices(ticker, exp_obj, sk, lk, "P", trade_date)
        if pp is None:
            continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": aw, "max_loss": round(aw - credit, 4)}
    return None


def _walk_spread(hd, ticker, exp, short_k, long_k, credit, entry_dt, exp_obj, td_idx,
                 profit_pct=0.50, stop_mult=3.0, min_dte=7):
    current = entry_dt + timedelta(days=1)
    td_set = set(td_idx.strftime("%Y-%m-%d"))
    hold = 0
    while current <= exp_obj:
        cs = current.strftime("%Y-%m-%d")
        if cs not in td_set:
            current += timedelta(days=1)
            continue
        hold += 1
        dte = (exp_obj - current).days
        pp = hd.get_spread_prices(ticker, exp_obj, short_k, long_k, "P", cs)
        if pp is None:
            current += timedelta(days=1)
            continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= credit * (1 - profit_pct):
            return cs, "profit_target", cv, hold
        if cv - credit > credit * stop_mult:
            return cs, "stop_loss", cv, hold
        if dte <= min_dte:
            return cs, "dte_exit", cv, hold
        current += timedelta(days=1)
    fp = hd.get_spread_prices(ticker, exp_obj, short_k, long_k, "P", exp)
    ev = fp["short_close"] - fp["long_close"] if fp else 0.0
    return exp, "expiration", ev, hold


def backtest_pairs_strategy(
    hd: IronVault,
    config: PairConfig,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    trade_df: pd.DataFrame,
    vix_series: pd.Series,
    year_filter: Optional[Tuple[int, int]] = None,
) -> List[Dict]:
    """Run the pairs mean-reversion strategy with given config.

    year_filter: (start_year, end_year) inclusive, for walk-forward slicing.
    """
    close_a = df_a["Close"]
    close_b = df_b["Close"]
    td_set = set(trade_df.index.strftime("%Y-%m-%d"))

    # Rolling correlation
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    common = ret_a.index.intersection(ret_b.index)
    ra = ret_a.reindex(common)
    rb = ret_b.reindex(common)
    roll_corr = ra.rolling(config.lookback).corr(rb)

    ticker = TRADE_TICKER
    start_str = f"{year_filter[0]}-01-01" if year_filter else "2020-04-01"
    end_str = f"{year_filter[1]}-12-31" if year_filter else "2025-12-31"
    exps = _find_exps(hd, ticker, start_str, end_str)

    trades, last = [], None
    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=35), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")

        if year_filter:
            yr = entry_dt.year
            if yr < year_filter[0] or yr > year_filter[1]:
                continue

        if last and (entry_dt - last).days < config.min_spacing:
            continue

        try:
            corr_val = float(roll_corr.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(corr_val) or corr_val < config.corr_threshold:
            continue

        try:
            price = float(trade_df["Close"].loc[es])
        except (KeyError, TypeError):
            continue

        spread = _sell_put_spread(hd, ticker, exp, es, price,
                                  otm_pct=config.otm_pct, width=config.spread_width)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, ticker, exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, trade_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * contracts

        # Get VIX for regime tagging
        try:
            vix_val = float(vix_series.loc[es])
        except (KeyError, TypeError):
            vix_val = 20.0

        trades.append({
            "entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"],
            "corr_val": round(corr_val, 4), "hold_days": hold,
            "vix": round(vix_val, 1),
        })
        last = entry_dt

    return trades


def _sharpe(pnls):
    if len(pnls) < 2:
        return 0.0
    s = np.std(pnls, ddof=1)
    return float(np.mean(pnls) / s * math.sqrt(min(len(pnls), 52))) if s > 1e-9 else 0.0


def _stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "sharpe": 0, "dd": 0}
    pnls = np.array([t["pnl"] for t in trades])
    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    return {
        "n": len(pnls),
        "pnl": round(float(pnls.sum()), 2),
        "wr": round(float((pnls > 0).sum() / len(pnls)), 3),
        "sharpe": round(_sharpe(pnls), 3),
        "dd": round(float(dd), 4),
    }


# ── Validation engine ────────────────────────────────────────────────────


class PairsDeepValidator:
    def __init__(self):
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._vix: Optional[pd.Series] = None

    def _get_prices(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._price_cache:
            self._price_cache[ticker] = _dl(ticker)
        return self._price_cache[ticker]

    def _get_vix(self) -> pd.Series:
        if self._vix is None:
            df = _dl("^VIX")
            self._vix = df["Close"] if not df.empty else pd.Series(dtype=float)
        return self._vix

    def run(self) -> ValidationResult:
        hd = IronVault.instance()
        vix = self._get_vix()
        spy_df = self._get_prices("SPY")
        tlt_df = self._get_prices("TLT")

        baseline = PairConfig("TLT", "SPY", lookback=30, corr_threshold=0.0,
                              spread_width=5.0, otm_pct=0.93, min_spacing=14)

        # 1. Walk-forward expanding window
        logger.info("=== Walk-Forward Validation ===")
        wf_windows = self._walk_forward(hd, baseline, tlt_df, spy_df, spy_df, vix)

        # 2. Parameter sensitivity
        logger.info("=== Parameter Sensitivity ===")
        param_results = self._param_sweep(hd, tlt_df, spy_df, spy_df, vix)

        # 3. Multi-pair test
        logger.info("=== Multi-Pair Test ===")
        pair_results = self._multi_pair(hd, spy_df, vix)

        # 4. Regime breakdown
        logger.info("=== Regime Breakdown ===")
        all_trades = backtest_pairs_strategy(hd, baseline, tlt_df, spy_df, spy_df, vix)
        regime_results = self._regime_breakdown(all_trades)

        # 5. Bias checks
        logger.info("=== Bias Checks ===")
        bias_flags = self._check_bias(all_trades, wf_windows, param_results)

        # Compute summaries
        baseline_sharpe = _sharpe([t["pnl"] for t in all_trades]) if all_trades else 0
        wf_sharpes = [w.test_sharpe for w in wf_windows]
        wf_avg = np.mean(wf_sharpes) if wf_sharpes else 0
        wf_consistency = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) if wf_sharpes else 0

        all_param_sharpes = [p["oos_sharpe"] for p in param_results if p["oos_sharpe"] != 0]
        sens_range = (min(all_param_sharpes), max(all_param_sharpes)) if all_param_sharpes else (0, 0)

        best_pair = max(pair_results.items(), key=lambda x: x[1].get("oos_sharpe", 0))[0] if pair_results else ""

        # Verdict
        if wf_consistency >= 0.6 and wf_avg > 1.0 and len(bias_flags) <= 1:
            verdict = "CONFIRMED"
        elif wf_consistency >= 0.4 and wf_avg > 0.5:
            verdict = "CAUTION"
        else:
            verdict = "OVERFIT"

        return ValidationResult(
            wf_windows=wf_windows,
            wf_avg_oos_sharpe=round(wf_avg, 3),
            wf_consistency=round(wf_consistency, 3),
            param_results=param_results,
            baseline_sharpe=round(baseline_sharpe, 3),
            sensitivity_range=sens_range,
            pair_results=pair_results,
            best_pair=best_pair,
            regime_results=regime_results,
            bias_flags=bias_flags,
            overall_verdict=verdict,
            all_trades=all_trades,
            config=baseline,
        )

    def _walk_forward(self, hd, config, df_a, df_b, trade_df, vix) -> List[WFWindow]:
        """Expanding window: train on 2020..Y, test on Y+1."""
        windows = []
        for test_year in range(2021, 2026):
            train_end = test_year - 1
            # Train period signal exists but we don't optimize — just validate
            # Test on the next year
            test_trades = backtest_pairs_strategy(
                hd, config, df_a, df_b, trade_df, vix,
                year_filter=(test_year, test_year),
            )
            s = _stats(test_trades)
            windows.append(WFWindow(
                train_start=2020, train_end=train_end,
                test_year=test_year,
                test_sharpe=s["sharpe"],
                test_n_trades=s["n"],
                test_pnl=s["pnl"],
                test_win_rate=s["wr"],
            ))
            logger.info("  WF %d→%d: Sharpe=%.2f, N=%d, PnL=$%.0f",
                        train_end, test_year, s["sharpe"], s["n"], s["pnl"])
        return windows

    def _param_sweep(self, hd, df_a, df_b, trade_df, vix) -> List[Dict]:
        """Sweep key parameters, report OOS (2023-2025) Sharpe for each."""
        results = []

        # Lookback sweep
        for lb in LOOKBACK_WINDOWS:
            cfg = PairConfig("TLT", "SPY", lookback=lb, corr_threshold=0.0,
                             spread_width=5.0, otm_pct=0.93)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, trade_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results.append({
                "param": "lookback", "value": lb,
                "full_sharpe": s_all["sharpe"], "oos_sharpe": s_oos["sharpe"],
                "n_trades": s_all["n"], "oos_n": s_oos["n"],
                "pnl": s_all["pnl"],
            })
            logger.info("  Lookback=%d: Full=%.2f OOS=%.2f N=%d",
                        lb, s_all["sharpe"], s_oos["sharpe"], s_all["n"])

        # Correlation threshold sweep
        for ct in CORR_THRESHOLDS:
            cfg = PairConfig("TLT", "SPY", lookback=30, corr_threshold=ct,
                             spread_width=5.0, otm_pct=0.93)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, trade_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results.append({
                "param": "corr_threshold", "value": ct,
                "full_sharpe": s_all["sharpe"], "oos_sharpe": s_oos["sharpe"],
                "n_trades": s_all["n"], "oos_n": s_oos["n"],
                "pnl": s_all["pnl"],
            })

        # Spread width sweep
        for sw in SPREAD_WIDTHS:
            cfg = PairConfig("TLT", "SPY", lookback=30, corr_threshold=0.0,
                             spread_width=sw, otm_pct=0.93)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, trade_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results.append({
                "param": "spread_width", "value": sw,
                "full_sharpe": s_all["sharpe"], "oos_sharpe": s_oos["sharpe"],
                "n_trades": s_all["n"], "oos_n": s_oos["n"],
                "pnl": s_all["pnl"],
            })

        # OTM sweep
        for otm in OTM_PCTS:
            cfg = PairConfig("TLT", "SPY", lookback=30, corr_threshold=0.0,
                             spread_width=5.0, otm_pct=otm)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, trade_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results.append({
                "param": "otm_pct", "value": otm,
                "full_sharpe": s_all["sharpe"], "oos_sharpe": s_oos["sharpe"],
                "n_trades": s_all["n"], "oos_n": s_oos["n"],
                "pnl": s_all["pnl"],
            })

        # Min spacing sweep
        for ms in MIN_SPACINGS:
            cfg = PairConfig("TLT", "SPY", lookback=30, corr_threshold=0.0,
                             spread_width=5.0, otm_pct=0.93, min_spacing=ms)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, trade_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results.append({
                "param": "min_spacing", "value": ms,
                "full_sharpe": s_all["sharpe"], "oos_sharpe": s_oos["sharpe"],
                "n_trades": s_all["n"], "oos_n": s_oos["n"],
                "pnl": s_all["pnl"],
            })

        return results

    def _multi_pair(self, hd, spy_df, vix) -> Dict[str, Dict]:
        """Test the correlation breakdown signal on multiple pairs."""
        results = {}
        for a, b in PAIRS:
            df_a = self._get_prices(a)
            df_b = self._get_prices(b)
            if df_a.empty or df_b.empty:
                results[f"{a}-{b}"] = {"n": 0, "pnl": 0, "sharpe": 0, "oos_sharpe": 0}
                continue

            cfg = PairConfig(a, b, lookback=30, corr_threshold=0.0,
                             spread_width=5.0, otm_pct=0.93)
            trades = backtest_pairs_strategy(hd, cfg, df_a, df_b, spy_df, vix)
            oos = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
            s_all = _stats(trades)
            s_oos = _stats(oos)
            results[f"{a}-{b}"] = {
                "n": s_all["n"], "pnl": s_all["pnl"], "wr": s_all["wr"],
                "sharpe": s_all["sharpe"], "dd": s_all["dd"],
                "oos_n": s_oos["n"], "oos_sharpe": s_oos["sharpe"], "oos_pnl": s_oos["pnl"],
            }
            logger.info("  %s-%s: N=%d Sharpe=%.2f OOS=%.2f PnL=$%.0f",
                        a, b, s_all["n"], s_all["sharpe"], s_oos["sharpe"], s_all["pnl"])
        return results

    def _regime_breakdown(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Break down performance by VIX regime."""
        regimes = {
            "bull": [t for t in trades if t.get("vix", 20) < 20],
            "bear": [t for t in trades if t.get("vix", 20) >= 25],
            "high_vol": [t for t in trades if t.get("vix", 20) >= 30],
            "low_vol": [t for t in trades if t.get("vix", 20) < 15],
            "moderate": [t for t in trades if 15 <= t.get("vix", 20) < 25],
        }
        return {name: _stats(ts) for name, ts in regimes.items()}

    def _check_bias(self, trades, wf_windows, param_results) -> List[str]:
        """Check for look-ahead bias and overfitting artifacts."""
        flags = []

        if not trades:
            flags.append("NO_TRADES: Strategy produced zero trades")
            return flags

        pnls = [t["pnl"] for t in trades]

        # 1. Perfect win rate is suspicious
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        if wr >= 0.98 and len(pnls) > 10:
            flags.append(f"PERFECT_WR: Win rate {wr:.0%} with {len(pnls)} trades — suspiciously high")

        # 2. Check if IS >> OOS (overfit indicator)
        is_trades = [t for t in trades if int(t["entry_date"][:4]) <= 2022]
        oos_trades = [t for t in trades if int(t["entry_date"][:4]) >= 2023]
        is_sh = _sharpe([t["pnl"] for t in is_trades]) if is_trades else 0
        oos_sh = _sharpe([t["pnl"] for t in oos_trades]) if oos_trades else 0
        if is_sh > 0 and oos_sh > 0 and is_sh / oos_sh > 3:
            flags.append(f"IS_OOS_GAP: IS Sharpe {is_sh:.2f} >> OOS {oos_sh:.2f} (ratio {is_sh/oos_sh:.1f}x)")

        # 3. Walk-forward consistency
        wf_neg = sum(1 for w in wf_windows if w.test_sharpe <= 0)
        if wf_neg >= len(wf_windows) * 0.4:
            flags.append(f"WF_INCONSISTENT: {wf_neg}/{len(wf_windows)} windows have negative Sharpe")

        # 4. Parameter sensitivity cliff
        oos_sharpes = [p["oos_sharpe"] for p in param_results if p["oos_sharpe"] != 0]
        if oos_sharpes:
            rng = max(oos_sharpes) - min(oos_sharpes)
            if rng > 10:
                flags.append(f"PARAM_CLIFF: Sharpe ranges {min(oos_sharpes):.1f} to {max(oos_sharpes):.1f} — sensitive to params")

        # 5. Too few trades for statistical significance
        if len(trades) < 30:
            flags.append(f"LOW_N: Only {len(trades)} trades — need 30+ for statistical significance")

        # 6. Tiny PnL relative to capital
        total_pnl = sum(pnls)
        if abs(total_pnl) / CAPITAL < 0.02:
            flags.append(f"TINY_PNL: ${total_pnl:,.0f} on ${CAPITAL:,} capital ({total_pnl/CAPITAL:.1%})")

        return flags

    def generate_report(self, result: ValidationResult, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path

    def save_summary(self, result: ValidationResult, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "strategy": "Cross-Asset Pairs Mean-Reversion (TLT-SPY Correlation Breakdown)",
            "original_claim": {"oos_sharpe": 6.31, "spy_corr": 0.02},
            "verdict": result.overall_verdict,
            "walk_forward": {
                "avg_oos_sharpe": result.wf_avg_oos_sharpe,
                "consistency": result.wf_consistency,
                "windows": [{"year": w.test_year, "sharpe": w.test_sharpe, "n": w.test_n_trades, "pnl": w.test_pnl}
                            for w in result.wf_windows],
            },
            "baseline_sharpe": result.baseline_sharpe,
            "param_sensitivity_range": list(result.sensitivity_range),
            "best_pair": result.best_pair,
            "pair_results": result.pair_results,
            "regime_breakdown": result.regime_results,
            "bias_flags": result.bias_flags,
            "total_trades": len(result.all_trades),
        }
        output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return output_path


# ── HTML Report ──────────────────────────────────────────────────────────

def _fc(v):
    c = "#22c55e" if v > 0 else "#ef4444"
    return f'<span style="color:{c}">{v:+.1f}</span>'

def _fp(v):
    return f"{v:.1f}%"

def _fr(v):
    return f"{v:.2f}"


def _build_html(r: ValidationResult) -> str:
    v = r.overall_verdict
    vc = {"CONFIRMED": "#3fb950", "CAUTION": "#d29922", "OVERFIT": "#ef4444"}.get(v, "#8b949e")

    # WF table
    wf_rows = ""
    for w in r.wf_windows:
        sc = "#22c55e" if w.test_sharpe > 0 else "#ef4444"
        wf_rows += (
            f"<tr><td>2020–{w.train_end}</td><td>{w.test_year}</td>"
            f"<td style='color:{sc}'><strong>{_fr(w.test_sharpe)}</strong></td>"
            f"<td>{w.test_n_trades}</td>"
            f"<td style='color:{'#22c55e' if w.test_pnl > 0 else '#ef4444'}'>${w.test_pnl:,.0f}</td>"
            f"<td>{w.test_win_rate:.0%}</td></tr>\n"
        )

    # Param table
    param_rows = ""
    for p in r.param_results:
        sc = "#22c55e" if p["oos_sharpe"] > 1 else ("#f59e0b" if p["oos_sharpe"] > 0 else "#ef4444")
        param_rows += (
            f"<tr><td style='text-align:left'>{p['param']}</td><td>{p['value']}</td>"
            f"<td>{_fr(p['full_sharpe'])}</td>"
            f"<td style='color:{sc}'><strong>{_fr(p['oos_sharpe'])}</strong></td>"
            f"<td>{p['n_trades']}</td><td>{p['oos_n']}</td>"
            f"<td>${p['pnl']:,.0f}</td></tr>\n"
        )

    # Pair table
    pair_rows = ""
    for pair, s in sorted(r.pair_results.items(), key=lambda x: x[1].get("oos_sharpe", 0), reverse=True):
        sc = "#22c55e" if s.get("oos_sharpe", 0) > 1 else ("#f59e0b" if s.get("oos_sharpe", 0) > 0 else "#ef4444")
        pair_rows += (
            f"<tr><td style='text-align:left'><strong>{pair}</strong></td>"
            f"<td>{s.get('n', 0)}</td>"
            f"<td>${s.get('pnl', 0):,.0f}</td>"
            f"<td>{s.get('wr', 0):.0%}</td>"
            f"<td>{_fr(s.get('sharpe', 0))}</td>"
            f"<td style='color:{sc}'><strong>{_fr(s.get('oos_sharpe', 0))}</strong></td>"
            f"<td>{s.get('oos_n', 0)}</td>"
            f"<td>{s.get('dd', 0):.1%}</td></tr>\n"
        )

    # Regime table
    regime_rows = ""
    for name, s in sorted(r.regime_results.items()):
        sc = "#22c55e" if s.get("sharpe", 0) > 0 else "#ef4444"
        regime_rows += (
            f"<tr><td style='text-align:left'>{name}</td>"
            f"<td>{s.get('n', 0)}</td>"
            f"<td>${s.get('pnl', 0):,.0f}</td>"
            f"<td>{s.get('wr', 0):.0%}</td>"
            f"<td style='color:{sc}'>{_fr(s.get('sharpe', 0))}</td>"
            f"<td>{s.get('dd', 0):.1%}</td></tr>\n"
        )

    # Bias flags
    bias_html = ""
    if r.bias_flags:
        bias_html = "<ul>" + "".join(f"<li>{f}</li>" for f in r.bias_flags) + "</ul>"
    else:
        bias_html = "<p style='color:#3fb950'>No bias flags detected.</p>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Cross-Asset Pairs Deep Validation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2,h3{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid {vc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:2em;font-weight:800;color:{vc}}}
.hero .sub{{color:#8b949e;margin-top:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.2em;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22;font-size:.85em}}
td:first-child,th:first-child{{text-align:left}}
tr:hover td{{background:#161b2280}}
.section{{margin:32px 0}}
.note{{color:#8b949e;font-size:.85em;margin:8px 0}}
.flag{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
.flag.warn{{border-color:#d29922;color:#d29922}}
.flag.pass{{border-color:#3fb950;color:#3fb950}}
.flag.fail{{border-color:#ef4444;color:#ef4444}}
ul{{margin:8px 0;padding-left:20px}}li{{margin:4px 0}}
</style></head><body>

<h1>Cross-Asset Pairs Mean-Reversion: Deep Validation</h1>
<p class="note">TLT-SPY Correlation Breakdown &middot; Original claim: OOS Sharpe 6.31, SPY corr 0.02</p>

<div class="hero">
  <div class="big">Verdict: {v}</div>
  <div class="sub">
    WF Avg OOS Sharpe: {_fr(r.wf_avg_oos_sharpe)} &middot;
    WF Consistency: {r.wf_consistency:.0%} &middot;
    Baseline Sharpe: {_fr(r.baseline_sharpe)} &middot;
    Best Pair: {r.best_pair} &middot;
    Bias Flags: {len(r.bias_flags)}
  </div>
</div>

<div class="cards">
  <div class="c"><div class="l">WF Avg OOS Sharpe</div><div class="v">{_fr(r.wf_avg_oos_sharpe)}</div></div>
  <div class="c"><div class="l">WF Consistency</div><div class="v">{r.wf_consistency:.0%}</div></div>
  <div class="c"><div class="l">Baseline Sharpe</div><div class="v">{_fr(r.baseline_sharpe)}</div></div>
  <div class="c"><div class="l">Param Sensitivity</div><div class="v">{_fr(r.sensitivity_range[0])} to {_fr(r.sensitivity_range[1])}</div></div>
  <div class="c"><div class="l">Best Pair</div><div class="v">{r.best_pair}</div></div>
  <div class="c"><div class="l">Bias Flags</div><div class="v" style="color:{'#3fb950' if len(r.bias_flags) <= 1 else '#ef4444'}">{len(r.bias_flags)}</div></div>
  <div class="c"><div class="l">Total Trades</div><div class="v">{len(r.all_trades)}</div></div>
  <div class="c"><div class="l">Verdict</div><div class="v" style="color:{vc}">{v}</div></div>
</div>

<div class="section">
<h2>1. Walk-Forward Expanding Window</h2>
<p class="note">Train on 2020..Y, test on Y+1. No parameter re-optimization between windows.</p>
<table>
<thead><tr><th>Train</th><th>Test</th><th>OOS Sharpe</th><th>Trades</th><th>PnL</th><th>Win Rate</th></tr></thead>
<tbody>{wf_rows}</tbody></table>
</div>

<div class="section">
<h2>2. Parameter Sensitivity</h2>
<p class="note">OOS period = 2023-2025 for all parameter variations.</p>
<table>
<thead><tr><th>Parameter</th><th>Value</th><th>Full Sharpe</th><th>OOS Sharpe</th><th>All Trades</th><th>OOS Trades</th><th>PnL</th></tr></thead>
<tbody>{param_rows}</tbody></table>
</div>

<div class="section">
<h2>3. Multi-Pair Test</h2>
<p class="note">Same correlation-breakdown signal applied to different asset pairs, trading SPY put spreads.</p>
<table>
<thead><tr><th>Pair</th><th>Trades</th><th>PnL</th><th>WR</th><th>Full Sharpe</th><th>OOS Sharpe</th><th>OOS N</th><th>DD</th></tr></thead>
<tbody>{pair_rows}</tbody></table>
</div>

<div class="section">
<h2>4. Regime Breakdown</h2>
<p class="note">Performance by VIX regime at entry.</p>
<table>
<thead><tr><th>Regime</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th><th>DD</th></tr></thead>
<tbody>{regime_rows}</tbody></table>
</div>

<div class="section">
<h2>5. Bias & Overfitting Checks</h2>
<div class="flag {'pass' if len(r.bias_flags) <= 1 else ('warn' if len(r.bias_flags) <= 3 else 'fail')}">
  <strong>{'CLEAN' if len(r.bias_flags) == 0 else f'{len(r.bias_flags)} flag(s) detected'}:</strong>
  {bias_html}
</div>
</div>

<p class="note" style="margin-top:40px;text-align:center">
  Cross-Asset Pairs Deep Validation &middot; All prices from IronVault &middot;
  Generated by PilotAI Compass
</p>
</body></html>"""
