"""Dispersion trading engine — captures the correlation risk premium by selling
index volatility and buying component volatility.

Core idea: implied correlation (from index IV vs component IVs) is
systematically higher than realised correlation. Selling SPY strangles
while buying component strangles profits when this spread compresses.

Provides:
  1. Implied correlation calculator
  2. Dispersion entry signal (implied > realised by threshold)
  3. Position sizing by vega exposure
  4. P&L attribution (correlation move vs vol move)
  5. Historical backtest of SPY vs top-10 components
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Constants ───────────────────────────────────────────────────────────────
TOP_COMPONENTS = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
                  "META", "BRK.B", "JPM", "UNH", "V"]

DEFAULT_ENTRY_SPREAD = 0.10     # enter when impl corr > real corr by 10pp
DEFAULT_EXIT_SPREAD = 0.02      # exit when spread compresses to 2pp
DEFAULT_MAX_VEGA = 500.0        # max portfolio vega exposure ($)
DEFAULT_LOOKBACK = 60           # days for realised correlation
TRADING_DAYS = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class CorrelationSnapshot:
    """Implied vs realised correlation at one point in time."""
    date: str
    implied_corr: float        # from index IV vs component IVs
    realised_corr: float       # from rolling returns
    spread: float              # implied - realised
    regime: str                # "risk_on", "risk_off", "transition"


@dataclass
class DispersionSignal:
    """Entry/exit signal for dispersion trade."""
    date: str
    action: str                # "enter", "exit", "hold", "none"
    spread: float
    implied_corr: float
    realised_corr: float
    confidence: float          # 0-1


@dataclass
class VegaSizing:
    """Vega-balanced position sizing."""
    index_vega: float          # vega from short index strangles
    component_vega: float      # vega from long component strangles
    net_vega: float            # target ~0 (vega-neutral)
    index_contracts: int
    component_contracts: Dict[str, int]


@dataclass
class PnLAttribution:
    """Decompose P&L into correlation and vol components."""
    total_pnl: float
    correlation_pnl: float     # from corr spread compressing
    vol_pnl: float             # from overall vol level change
    theta_pnl: float           # time decay
    residual_pnl: float


@dataclass
class DispersionTrade:
    """Single dispersion trade record."""
    entry_date: str
    exit_date: str
    entry_spread: float
    exit_spread: float
    hold_days: int
    pnl: float
    attribution: PnLAttribution
    index_contracts: int
    sizing: Optional[VegaSizing] = None


@dataclass
class DispersionResult:
    """Complete backtest output."""
    trades: List[DispersionTrade] = field(default_factory=list)
    correlation_history: List[CorrelationSnapshot] = field(default_factory=list)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
    avg_entry_spread: float = 0.0
    ending_capital: float = 0.0
    generated_at: str = ""


# ── Implied correlation calculator ──────────────────────────────────────────
def implied_correlation(
    index_iv: float,
    component_ivs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute implied correlation from index IV and component IVs.

    Formula: ρ_impl = (σ_index² - Σ wᵢ² σᵢ²) / (Σᵢ≠ⱼ wᵢwⱼσᵢσⱼ)

    Simplified for equal weights:
      ρ = (σ_index² - (1/n)Σσᵢ²) / ((1-1/n) × mean(σᵢ)²)
    """
    n = len(component_ivs)
    if n < 2 or index_iv <= 0:
        return 0.0

    if weights is None:
        weights = np.ones(n) / n

    var_index = index_iv ** 2
    weighted_var_sum = np.sum(weights ** 2 * component_ivs ** 2)
    cross_term = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cross_term += 2 * weights[i] * weights[j] * component_ivs[i] * component_ivs[j]

    if cross_term <= 1e-12:
        return 0.0

    rho = (var_index - weighted_var_sum) / cross_term
    return float(np.clip(rho, -1.0, 1.0))


def realised_correlation(
    index_returns: np.ndarray,
    component_returns: np.ndarray,
) -> float:
    """Average pairwise realised correlation from return arrays.

    component_returns: (T, N) array.
    """
    n = component_returns.shape[1] if component_returns.ndim > 1 else 1
    if n < 2 or len(index_returns) < 10:
        return 0.0

    corr_matrix = np.corrcoef(component_returns.T)
    # Average off-diagonal
    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(np.mean(corr_matrix[mask]))
    return float(np.clip(avg_corr, -1.0, 1.0))


# ── Correlation regime classifier ───────────────────────────────────────────
def classify_correlation_regime(
    implied_corr: float,
    realised_corr: float,
) -> str:
    """Classify current correlation regime.

    risk_off: both correlations high (>0.6) — everything moving together
    risk_on: both low (<0.4) — dispersion, stock-picking works
    transition: mixed signals
    """
    avg = (implied_corr + realised_corr) / 2
    if avg > 0.6:
        return "risk_off"
    if avg < 0.4:
        return "risk_on"
    return "transition"


# ── Vega-balanced sizing ────────────────────────────────────────────────────
def compute_vega_sizing(
    index_iv: float,
    component_ivs: Dict[str, float],
    spy_price: float,
    max_vega: float = DEFAULT_MAX_VEGA,
    dte: float = 30.0,
) -> VegaSizing:
    """Size positions to be approximately vega-neutral.

    Sell index strangles (short vega), buy component strangles (long vega).
    """
    # Index vega per contract ≈ S × √T × 0.01 / 100
    sqrt_t = math.sqrt(max(dte / 365, 0.001))
    index_vega_per = spy_price * sqrt_t * 0.01

    if index_vega_per <= 0:
        return VegaSizing(0, 0, 0, 0, {})

    # Target: short X contracts of index, long proportional components
    index_contracts = max(1, int(max_vega / index_vega_per))
    total_index_vega = index_contracts * index_vega_per

    # Distribute long vega across components equally
    n_comp = len(component_ivs)
    if n_comp == 0:
        return VegaSizing(total_index_vega, 0, total_index_vega, index_contracts, {})

    target_per_component = total_index_vega / n_comp
    comp_contracts: Dict[str, int] = {}
    total_comp_vega = 0.0

    for sym, iv in component_ivs.items():
        comp_price = spy_price * 0.5  # rough estimate
        comp_vega_per = comp_price * sqrt_t * 0.01
        if comp_vega_per > 0:
            contracts = max(1, int(target_per_component / comp_vega_per))
        else:
            contracts = 1
        comp_contracts[sym] = contracts
        total_comp_vega += contracts * comp_vega_per

    return VegaSizing(
        index_vega=round(total_index_vega, 2),
        component_vega=round(total_comp_vega, 2),
        net_vega=round(total_comp_vega - total_index_vega, 2),
        index_contracts=index_contracts,
        component_contracts=comp_contracts,
    )


# ── P&L attribution ────────────────────────────────────────────────────────
def attribute_pnl(
    entry_spread: float,
    exit_spread: float,
    entry_vol: float,
    exit_vol: float,
    hold_days: int,
    notional: float,
) -> PnLAttribution:
    """Decompose dispersion trade P&L.

    correlation_pnl: profit from spread compression
    vol_pnl: profit/loss from overall vol level change
    theta_pnl: time decay (always positive for short straddles)
    """
    spread_change = entry_spread - exit_spread  # positive if spread compressed
    vol_change = exit_vol - entry_vol

    # Correlation P&L: spread change × notional × sensitivity
    corr_pnl = spread_change * notional * 2.0  # ~2x notional sensitivity

    # Vol P&L: short index vol, long component vol → net ~0 if vega-neutral
    # But residual vega from imperfect hedge
    vol_pnl = -vol_change * notional * 0.1  # small residual

    # Theta: positive, proportional to hold time
    theta_daily = notional * 0.002  # ~0.2% per day
    theta_pnl = theta_daily * hold_days

    total = corr_pnl + vol_pnl + theta_pnl
    residual = 0.0  # would capture gamma, skew, etc.

    return PnLAttribution(
        total_pnl=round(total, 2),
        correlation_pnl=round(corr_pnl, 2),
        vol_pnl=round(vol_pnl, 2),
        theta_pnl=round(theta_pnl, 2),
        residual_pnl=round(residual, 2),
    )


# ── Backtest engine ─────────────────────────────────────────────────────────
class DispersionBacktest:
    """Backtest dispersion strategy on synthetic data."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        entry_spread: float = DEFAULT_ENTRY_SPREAD,
        exit_spread: float = DEFAULT_EXIT_SPREAD,
        max_vega: float = DEFAULT_MAX_VEGA,
        lookback: int = DEFAULT_LOOKBACK,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.entry_threshold = entry_spread
        self.exit_threshold = exit_spread
        self.max_vega = max_vega
        self.lookback = lookback
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        index_returns: pd.Series,
        component_returns: pd.DataFrame,
        index_iv: pd.Series,
        component_ivs: pd.DataFrame,
    ) -> DispersionResult:
        """Run dispersion backtest.

        All inputs indexed by date, aligned.
        """
        n = len(index_returns)
        if n < self.lookback + 20:
            return DispersionResult(generated_at=_now())

        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        trades: List[DispersionTrade] = []
        corr_history: List[CorrelationSnapshot] = []
        daily_pnl: List[float] = []

        in_trade = False
        trade_entry_idx = 0
        trade_entry_spread = 0.0
        trade_entry_vol = 0.0

        for i in range(self.lookback, n):
            # Rolling realised correlation
            window = component_returns.iloc[i - self.lookback:i].values
            real_corr = realised_correlation(
                index_returns.iloc[i - self.lookback:i].values, window,
            )

            # Implied correlation
            idx_iv = float(index_iv.iloc[i])
            comp_iv_vals = component_ivs.iloc[i].values.astype(float)
            impl_corr = implied_correlation(idx_iv, comp_iv_vals)

            spread = impl_corr - real_corr
            regime = classify_correlation_regime(impl_corr, real_corr)

            corr_history.append(CorrelationSnapshot(
                date=str(index_returns.index[i]),
                implied_corr=round(impl_corr, 4),
                realised_corr=round(real_corr, 4),
                spread=round(spread, 4),
                regime=regime,
            ))

            if not in_trade:
                # Entry signal
                if spread >= self.entry_threshold and regime != "risk_off":
                    in_trade = True
                    trade_entry_idx = i
                    trade_entry_spread = spread
                    trade_entry_vol = idx_iv
            else:
                # Exit conditions
                should_exit = (
                    spread <= self.exit_threshold  # spread compressed
                    or i - trade_entry_idx >= 30   # max hold 30 days
                    or regime == "risk_off"         # risk-off → close
                )

                if should_exit:
                    hold = i - trade_entry_idx
                    notional = capital * 0.10  # 10% of capital per trade

                    attr = attribute_pnl(
                        trade_entry_spread, spread,
                        trade_entry_vol, idx_iv,
                        hold, notional,
                    )
                    # Add noise
                    noise = self.rng.randn() * notional * 0.005
                    pnl = attr.total_pnl + noise

                    sizing = compute_vega_sizing(
                        idx_iv,
                        {c: float(component_ivs.iloc[i][c]) for c in component_ivs.columns},
                        float(index_returns.index[i].year * 2 + 300) if hasattr(index_returns.index[i], 'year') else 450.0,
                        self.max_vega,
                    )

                    trades.append(DispersionTrade(
                        entry_date=str(index_returns.index[trade_entry_idx]),
                        exit_date=str(index_returns.index[i]),
                        entry_spread=round(trade_entry_spread, 4),
                        exit_spread=round(spread, 4),
                        hold_days=hold,
                        pnl=round(pnl, 2),
                        attribution=attr,
                        index_contracts=sizing.index_contracts,
                        sizing=sizing,
                    ))

                    capital += pnl
                    daily_pnl.append(pnl)
                    in_trade = False

                    if capital > peak:
                        peak = capital
                    dd = (peak - capital) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)

        # Metrics
        executed = trades
        wins = sum(1 for t in executed if t.pnl > 0)
        total_pnl = sum(t.pnl for t in executed)
        total_return = (capital - self.starting_capital) / self.starting_capital * 100
        years = (n - self.lookback) / TRADING_DAYS
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(len(dr) / max(years, 0.1))) if dr.std() > 0 else 0

        win_sum = sum(t.pnl for t in executed if t.pnl > 0)
        loss_sum = abs(sum(t.pnl for t in executed if t.pnl < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 0

        return DispersionResult(
            trades=trades,
            correlation_history=corr_history,
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            win_rate_pct=round(wins / len(executed) * 100, 1) if executed else 0,
            profit_factor=round(pf, 2),
            total_trades=len(executed),
            avg_hold_days=round(float(np.mean([t.hold_days for t in executed])), 1) if executed else 0,
            avg_entry_spread=round(float(np.mean([t.entry_spread for t in executed])), 4) if executed else 0,
            ending_capital=round(capital, 2),
            generated_at=_now(),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_dispersion_data(
    n: int = 1000, n_components: int = 10, seed: int = 42,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate synthetic index + component returns and IVs."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)

    # Common factor (market)
    market = rng.randn(n) * 0.01

    # Index returns: market + small noise
    index_ret = market + rng.randn(n) * 0.002
    index_series = pd.Series(index_ret, index=idx, name="SPY")

    # Component returns: market × beta + idiosyncratic
    comp_names = TOP_COMPONENTS[:n_components]
    comp_data = {}
    for j, name in enumerate(comp_names):
        beta = 0.8 + rng.rand() * 0.6  # 0.8-1.4
        idio = rng.randn(n) * 0.015
        comp_data[name] = market * beta + idio
    comp_df = pd.DataFrame(comp_data, index=idx)

    # IVs: VIX-like for index, slightly higher for components
    base_vix = 18 + np.cumsum(rng.randn(n) * 0.5) * 0
    for i in range(1, n):
        base_vix[i] = max(10, min(60, base_vix[i - 1] + 0.03 * (18 - base_vix[i - 1]) + rng.randn() * 1.0))
    base_vix[0] = 18

    index_iv = pd.Series(base_vix, index=idx, name="SPY_IV")

    comp_iv_data = {}
    for name in comp_names:
        # Component IVs = index IV × (1.1 to 1.5) + noise
        mult = 1.1 + rng.rand() * 0.4
        comp_iv_data[name] = base_vix * mult + rng.randn(n) * 2
    comp_iv_df = pd.DataFrame(comp_iv_data, index=idx)
    comp_iv_df = comp_iv_df.clip(lower=5)

    return index_series, comp_df, index_iv, comp_iv_df


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
