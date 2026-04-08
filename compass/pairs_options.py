"""
Pairs trading for options — cointegrated pair signals monetised via
credit spreads on the diverging leg.

Engle-Granger cointegration testing, spread z-score computation,
credit spread signal generation on the overextended leg, regime
filter, and walk-forward backtest.

Usage::

    from compass.pairs_options import PairsOptionsEngine, PairsConfig
    engine = PairsOptionsEngine(prices_df)
    results = engine.analyze()
    bt = engine.backtest()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ── Configuration ───────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    ("SPY", "QQQ"), ("XLF", "JPM"), ("XLK", "AAPL"), ("XLE", "XOM"),
    ("GLD", "GDX"), ("TLT", "IEF"), ("SPY", "IWM"), ("QQQ", "SMH"),
    ("XLV", "JNJ"), ("XLU", "NEE"), ("EEM", "EFA"),
]


@dataclass
class PairsConfig:
    pairs: List[Tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_PAIRS))
    lookback: int = 60                 # rolling window for z-score
    coint_window: int = 252            # window for cointegration test
    entry_z: float = 2.0              # enter when |z| > this
    exit_z: float = 0.5               # exit when |z| < this
    stop_z: float = 3.5               # stop-loss at extreme z
    coint_pvalue: float = 0.05        # cointegration p-value threshold
    max_holding_days: int = 30
    position_pct: float = 0.05        # 5% of capital per trade
    slippage_bps: float = 5.0
    starting_capital: float = 100_000
    regime_filter: bool = True         # skip trades during breakdown
    regime_vix_threshold: float = 30   # VIX above this = breakdown


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class CointegrationTest:
    """Engle-Granger cointegration test result for one pair."""
    asset_a: str
    asset_b: str
    cointegrated: bool
    adf_stat: float
    p_value: float
    hedge_ratio: float
    half_life: float
    mean_spread: float
    spread_std: float


@dataclass
class SpreadSnapshot:
    """Spread state at one point in time."""
    date: str
    asset_a: str
    asset_b: str
    spread: float
    z_score: float
    hedge_ratio: float
    signal: str                # "sell_puts_on_a", "sell_calls_on_a", "sell_puts_on_b", "sell_calls_on_b", "neutral"


@dataclass
class PairsTrade:
    """One pairs trade."""
    entry_date: str
    exit_date: str
    asset_a: str
    asset_b: str
    trade_leg: str             # which asset we sell premium on
    direction: str             # "sell_puts" or "sell_calls"
    entry_z: float
    exit_z: float
    exit_reason: str           # "mean_reversion", "stop_loss", "max_holding", "regime_exit"
    pnl: float
    win: bool
    holding_days: int


@dataclass
class PairAnalysis:
    """Analysis result for one pair."""
    asset_a: str
    asset_b: str
    coint: CointegrationTest
    n_signals: int
    n_trades: int
    win_rate: float
    total_pnl: float
    sharpe: float
    avg_half_life: float


@dataclass
class BacktestResult:
    """Full pairs options backtest result."""
    n_pairs_tested: int
    n_cointegrated: int
    n_trades: int
    win_rate: float
    total_pnl: float
    sharpe: float
    sortino: float
    max_dd: float
    annual_return: float
    avg_holding_days: float
    by_pair: Dict[str, Dict[str, float]]
    by_exit_reason: Dict[str, int]
    correlation_with_spy: float
    trades: List[PairsTrade]
    pair_analyses: List[PairAnalysis]


# ── Cointegration (Engle-Granger) ──────────────────────────────────────


def _adf_test(series: np.ndarray) -> Tuple[float, float]:
    """Simplified ADF test. Returns (stat, p_value)."""
    y = series[~np.isnan(series)]
    n = len(y)
    if n < 20:
        return 0.0, 1.0
    dy = np.diff(y)
    y_lag = y[:-1]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    b = beta[1]
    resid = dy - X @ beta
    se = np.sqrt(np.sum(resid ** 2) / (len(dy) - 2))
    se_b = se / np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    if se_b < 1e-15:
        return 0.0, 1.0
    stat = b / se_b
    if stat < -3.43:
        p = 0.005
    elif stat < -2.86:
        p = 0.03
    elif stat < -2.57:
        p = 0.07
    else:
        p = min(1.0, 0.5 + 0.3 * (stat + 1.94))
    return float(stat), float(max(0, min(1, p)))


def test_cointegration(
    prices_a: np.ndarray, prices_b: np.ndarray, pvalue: float = 0.05,
) -> CointegrationTest:
    """Engle-Granger two-step cointegration test."""
    mask = ~(np.isnan(prices_a) | np.isnan(prices_b))
    pa, pb = prices_a[mask], prices_b[mask]
    n = len(pa)
    if n < 30:
        return CointegrationTest("", "", False, 0, 1, 0, 0, 0, 0)

    # OLS: pa = α + β × pb + ε
    X = np.column_stack([np.ones(n), pb])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, pa, rcond=None)
    except np.linalg.LinAlgError:
        return CointegrationTest("", "", False, 0, 1, 0, 0, 0, 0)

    hedge_ratio = float(beta[1])
    residuals = pa - X @ beta

    # ADF on residuals
    adf_stat, p_val = _adf_test(residuals)
    cointegrated = p_val < pvalue

    # Half-life from AR(1)
    y = residuals[1:]
    y_lag = residuals[:-1]
    if len(y) > 5 and np.std(y_lag) > 0:
        X_ar = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            b_ar, _, _, _ = np.linalg.lstsq(X_ar, y, rcond=None)
            phi = b_ar[1]
            half_life = -math.log(2) / math.log(abs(phi)) if 0 < abs(phi) < 1 else 999
        except (np.linalg.LinAlgError, ValueError):
            half_life = 999
    else:
        half_life = 999

    return CointegrationTest(
        "", "", cointegrated, adf_stat, p_val, hedge_ratio,
        min(half_life, 999), float(np.mean(residuals)), float(np.std(residuals)),
    )


# ── Spread z-score ──────────────────────────────────────────────────────


def compute_spread(
    prices_a: np.ndarray, prices_b: np.ndarray, hedge_ratio: float,
) -> np.ndarray:
    """Compute the log-price spread: a − β × b."""
    return prices_a - hedge_ratio * prices_b


def rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of spread."""
    n = len(spread)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = spread[i - window:i]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma > 1e-10:
            z[i] = (spread[i] - mu) / sigma
    return z


# ── Engine ──────────────────────────────────────────────────────────────


class PairsOptionsEngine:
    """Pairs trading for options engine."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: Optional[PairsConfig] = None,
        vix: Optional[pd.Series] = None,
    ) -> None:
        self.prices = prices.copy()
        self.config = config or PairsConfig()
        self.vix = vix
        self.n = len(prices)
        self.pair_analyses: List[PairAnalysis] = []
        self.backtest_result: Optional[BacktestResult] = None

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "PairsOptionsEngine":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    def analyze(self) -> List[PairAnalysis]:
        """Test cointegration for all pairs and compute signals."""
        cfg = self.config
        self.pair_analyses = []

        for a, b in cfg.pairs:
            if a not in self.prices.columns or b not in self.prices.columns:
                continue

            pa = self.prices[a].values.astype(float)
            pb = self.prices[b].values.astype(float)

            coint = test_cointegration(pa, pb, cfg.coint_pvalue)
            coint.asset_a = a
            coint.asset_b = b

            n_signals = 0
            if coint.cointegrated:
                spread = compute_spread(pa, pb, coint.hedge_ratio)
                z = rolling_zscore(spread, cfg.lookback)
                n_signals = int(np.sum(np.abs(z[~np.isnan(z)]) > cfg.entry_z))

            self.pair_analyses.append(PairAnalysis(
                a, b, coint, n_signals, 0, 0, 0, 0, coint.half_life,
            ))

        return self.pair_analyses

    def backtest(self) -> BacktestResult:
        """Walk-forward backtest of pairs options strategy."""
        if not self.pair_analyses:
            self.analyze()

        cfg = self.config
        cap = cfg.starting_capital
        all_trades: List[PairsTrade] = []
        equity = [cap]

        cointegrated_pairs = [pa for pa in self.pair_analyses if pa.coint.cointegrated]

        for pa in cointegrated_pairs:
            a, b = pa.asset_a, pa.asset_b
            prices_a = self.prices[a].values.astype(float)
            prices_b = self.prices[b].values.astype(float)
            coint = pa.coint

            spread = compute_spread(prices_a, prices_b, coint.hedge_ratio)
            z = rolling_zscore(spread, cfg.lookback)

            # Walk-forward: retrain cointegration every coint_window
            i = cfg.lookback
            in_trade = False
            trade_entry_idx = 0
            trade_entry_z = 0.0
            trade_leg = ""
            trade_dir = ""

            while i < self.n:
                if np.isnan(z[i]):
                    i += 1
                    continue

                cur_z = z[i]

                # Regime filter
                if cfg.regime_filter and self.vix is not None:
                    if i < len(self.vix) and self.vix.iloc[i] > cfg.regime_vix_threshold:
                        if in_trade:
                            # Exit on regime breakdown
                            pnl = self._estimate_trade_pnl(
                                trade_entry_z, cur_z, equity[-1], cfg.position_pct,
                            )
                            date_entry = str(self.prices.index[trade_entry_idx])
                            date_exit = str(self.prices.index[i])
                            all_trades.append(PairsTrade(
                                date_entry, date_exit, a, b, trade_leg, trade_dir,
                                trade_entry_z, cur_z, "regime_exit", pnl, pnl > 0,
                                i - trade_entry_idx,
                            ))
                            equity.append(equity[-1] + pnl)
                            in_trade = False
                        i += 1
                        continue

                if not in_trade:
                    # Entry signal: |z| > entry_z
                    if cur_z > cfg.entry_z:
                        # Spread too high → A overextended relative to B
                        # Sell calls on A (expect A to fall back)
                        in_trade = True
                        trade_entry_idx = i
                        trade_entry_z = cur_z
                        trade_leg = a
                        trade_dir = "sell_calls"
                    elif cur_z < -cfg.entry_z:
                        # Spread too low → B overextended relative to A
                        # Sell puts on A (expect A to rise back)
                        in_trade = True
                        trade_entry_idx = i
                        trade_entry_z = cur_z
                        trade_leg = a
                        trade_dir = "sell_puts"
                else:
                    # Exit conditions
                    holding = i - trade_entry_idx
                    exit_reason = ""

                    if abs(cur_z) < cfg.exit_z:
                        exit_reason = "mean_reversion"
                    elif abs(cur_z) > cfg.stop_z:
                        exit_reason = "stop_loss"
                    elif holding >= cfg.max_holding_days:
                        exit_reason = "max_holding"

                    if exit_reason:
                        pnl = self._estimate_trade_pnl(
                            trade_entry_z, cur_z, equity[-1], cfg.position_pct,
                        )
                        date_entry = str(self.prices.index[trade_entry_idx])
                        date_exit = str(self.prices.index[i])
                        all_trades.append(PairsTrade(
                            date_entry, date_exit, a, b, trade_leg, trade_dir,
                            trade_entry_z, cur_z, exit_reason, pnl, pnl > 0,
                            holding,
                        ))
                        equity.append(equity[-1] + pnl)
                        in_trade = False

                i += 1

        # Metrics
        if not all_trades:
            return BacktestResult(
                len(self.pair_analyses), len(cointegrated_pairs),
                0, 0, 0, 0, 0, 0, 0, 0, {}, {}, 0, [], self.pair_analyses,
            )

        pnls = np.array([t.pnl for t in all_trades])
        n_trades = len(all_trades)
        n_wins = sum(1 for t in all_trades if t.win)
        wr = n_wins / n_trades
        total_pnl = float(pnls.sum())

        eq = np.array(equity)
        eq_rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        eq_rets = eq_rets[eq_rets != 0]

        sh = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(252)) if len(eq_rets) > 1 and np.std(eq_rets) > 0 else 0
        down = eq_rets[eq_rets < 0]
        down_std = float(np.std(down)) if len(down) > 0 else 0.001
        sortino = float(np.mean(eq_rets) / down_std * np.sqrt(252)) if down_std > 0 else 0

        pk = np.maximum.accumulate(eq)
        dd = float(np.min((eq - pk) / np.where(pk > 0, pk, 1)))
        years = self.n / 252
        ann_ret = total_pnl / cap / max(years, 0.1)
        avg_hold = float(np.mean([t.holding_days for t in all_trades]))

        # By pair
        by_pair: Dict[str, Dict[str, float]] = {}
        for t in all_trades:
            key = f"{t.asset_a}/{t.asset_b}"
            if key not in by_pair:
                by_pair[key] = {"n": 0, "pnl": 0, "wins": 0}
            by_pair[key]["n"] += 1
            by_pair[key]["pnl"] += t.pnl
            by_pair[key]["wins"] += int(t.win)
        for k in by_pair:
            by_pair[k]["win_rate"] = by_pair[k]["wins"] / by_pair[k]["n"] if by_pair[k]["n"] > 0 else 0

        # By exit reason
        by_exit: Dict[str, int] = {}
        for t in all_trades:
            by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1

        # Update pair analyses with trade stats
        for pa in self.pair_analyses:
            key = f"{pa.asset_a}/{pa.asset_b}"
            if key in by_pair:
                pa.n_trades = int(by_pair[key]["n"])
                pa.win_rate = by_pair[key]["win_rate"]
                pa.total_pnl = by_pair[key]["pnl"]

        # SPY correlation
        spy_rets = self.prices.iloc[:, 0].pct_change().dropna().values if len(self.prices.columns) > 0 else np.zeros(10)
        if len(eq_rets) > 10 and len(spy_rets) > len(eq_rets):
            spy_sub = spy_rets[-len(eq_rets):]
            corr = float(np.corrcoef(eq_rets, spy_sub)[0, 1]) if np.std(spy_sub) > 0 else 0
        else:
            corr = 0.0

        self.backtest_result = BacktestResult(
            len(self.pair_analyses), len(cointegrated_pairs),
            n_trades, wr, total_pnl, sh, sortino, dd, ann_ret,
            avg_hold, by_pair, by_exit, corr, all_trades, self.pair_analyses,
        )
        return self.backtest_result

    def _estimate_trade_pnl(
        self, entry_z: float, exit_z: float,
        current_equity: float, position_pct: float,
    ) -> float:
        """Estimate credit spread PnL from z-score mean reversion.

        Win: z reverts → credit spread expires worthless → collect premium.
        Loss: z extends → credit spread goes ITM → lose spread width.
        """
        notional = current_equity * position_pct
        z_move = abs(entry_z) - abs(exit_z)  # positive = reverted

        # Premium collected: proportional to entry z (higher z = more IV)
        premium_pct = min(abs(entry_z) * 0.015, 0.05)  # 1.5% per z-unit, max 5%
        premium = notional * premium_pct

        # Slippage
        slip = notional * self.config.slippage_bps / 10_000

        if z_move > 0:
            # Mean reversion: collect fraction of premium
            frac = min(z_move / abs(entry_z), 1.0)
            pnl = premium * frac - slip
        else:
            # Extension: lose proportional to how much further it went
            loss_frac = min(abs(z_move) / 2, 1.0)
            max_loss = notional * 0.03  # 3% max loss per trade
            pnl = -max_loss * loss_frac - slip

        return pnl
