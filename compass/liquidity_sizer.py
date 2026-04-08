"""
Liquidity-aware position sizing engine.

Adapts trade size to real-time market liquidity to minimise slippage
and maximise capacity.

Components:
  1. Open interest / volume analysis
  2. Bid-ask spread monitoring across strikes and expirations
  3. Market impact estimation (order size → price impact)
  4. Capacity calculation (max contracts before >1% slippage)
  5. Adaptive sizing (scale with liquidity conditions)
  6. Roll optimisation (cheapest path considering liquidity)
  7. Backtest: liquidity-aware vs fixed sizing

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
CONTRACT_MULT = 100


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrikeLiquidity:
    """Liquidity snapshot for one strike/expiry."""
    strike: float
    expiry_days: int
    open_interest: int
    daily_volume: int
    bid_ask_spread: float      # dollars
    relative_spread: float     # spread / midprice
    midprice: float = 0.0


@dataclass
class LiquidityScore:
    """Aggregate liquidity assessment."""
    score: float               # 0-1 (1 = very liquid)
    avg_spread: float
    avg_oi: float
    avg_volume: float
    constraint: str            # "none" | "spread" | "volume" | "oi"


@dataclass
class ImpactEstimate:
    """Market impact for a given order."""
    contracts: int
    participation_rate: float  # contracts / daily_volume
    spread_cost_per: float     # per-contract spread cost
    impact_bps: float          # price impact in bps
    total_cost: float          # spread + impact, dollars


@dataclass
class CapacityResult:
    """Maximum position before slippage exceeds threshold."""
    max_contracts: int
    max_notional: float
    slippage_at_max: float     # slippage % at max size
    limiting_factor: str       # "spread" | "volume" | "oi" | "impact"


@dataclass
class AdaptiveSize:
    """Liquidity-adjusted position size."""
    base_contracts: int
    adjusted_contracts: int
    liquidity_score: float
    scale_factor: float
    reason: str


@dataclass
class RollPath:
    """Optimal roll from current to target expiry."""
    current_strike: float
    current_dte: int
    target_strike: float
    target_dte: int
    roll_cost: float           # net debit/credit for the roll
    liquidity_score: float     # of the target
    is_optimal: bool


@dataclass
class SizingBacktestResult:
    """Comparison of fixed vs liquidity-aware sizing."""
    fixed_return: float
    fixed_sharpe: float
    fixed_max_dd: float
    fixed_total_slippage: float
    adaptive_return: float
    adaptive_sharpe: float
    adaptive_max_dd: float
    adaptive_total_slippage: float
    slippage_reduction: float  # (fixed - adaptive) / fixed
    n_trades: int


# ---------------------------------------------------------------------------
# Synthetic liquidity data generator
# ---------------------------------------------------------------------------

def generate_option_chain(
    underlying: float = 450.0,
    n_strikes: int = 21,
    expiries: Tuple[int, ...] = (7, 14, 21, 30, 45),
    vix: float = 20.0,
    seed: int = 42,
) -> List[StrikeLiquidity]:
    """Generate realistic option chain liquidity data.

    Calibrated to SPY options:
    - ATM OI: 50K-200K, volume: 10K-50K
    - OTM 5%: OI drops 5-10x, spread widens 3-5x
    - Shorter DTE: higher volume, tighter spreads
    - Higher VIX: wider spreads, more volume
    """
    rng = np.random.default_rng(seed)
    strikes = np.linspace(underlying * 0.90, underlying * 1.10, n_strikes)
    chain: List[StrikeLiquidity] = []

    for dte in expiries:
        for k in strikes:
            moneyness = abs(k / underlying - 1.0)

            # OI: peaks at ATM, decays with moneyness
            base_oi = 150_000 * math.exp(-20 * moneyness)
            dte_mult = max(0.3, 1.0 - 0.015 * dte)  # shorter = more OI
            oi = int(base_oi * dte_mult * rng.uniform(0.7, 1.3))

            # Volume: fraction of OI, higher for short DTE
            vol_frac = 0.15 + 0.10 * (30 / max(dte, 1))
            volume = int(oi * vol_frac * rng.uniform(0.5, 1.5))

            # Bid-ask: tighter at ATM, wider OTM, wider with high VIX
            base_spread = 0.03
            moneyness_mult = 1.0 + 8.0 * moneyness ** 1.3
            vix_mult = 1.0 + 0.5 * max(0, (vix - 15) / 15)
            dte_spread = 0.8 + 0.4 * (dte / 30)
            spread = base_spread * moneyness_mult * vix_mult * dte_spread
            spread *= rng.uniform(0.8, 1.2)

            # Midprice (Black-Scholes-ish approximation)
            iv = vix / 100
            T = dte / TRADING_DAYS
            mid = max(0.05, iv * underlying * math.sqrt(T) * 0.4 *
                       math.exp(-5 * moneyness))

            chain.append(StrikeLiquidity(
                strike=float(k), expiry_days=dte,
                open_interest=max(100, oi),
                daily_volume=max(10, volume),
                bid_ask_spread=round(spread, 4),
                relative_spread=spread / max(mid, 0.01),
                midprice=round(mid, 4),
            ))
    return chain


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class LiquiditySizer:
    """Liquidity-aware position sizing engine.

    Args:
        max_participation: Max fraction of daily volume per trade.
        max_slippage_pct: Max acceptable slippage as % of premium.
        min_oi: Minimum open interest to consider a strike.
    """

    def __init__(
        self,
        max_participation: float = 0.02,
        max_slippage_pct: float = 0.01,
        min_oi: int = 500,
    ) -> None:
        self.max_participation = max_participation
        self.max_slippage_pct = max_slippage_pct
        self.min_oi = min_oi

    # ------------------------------------------------------------------
    # 1. OI / volume analysis
    # ------------------------------------------------------------------

    @staticmethod
    def analyse_chain(chain: List[StrikeLiquidity]) -> pd.DataFrame:
        """Convert chain to DataFrame for analysis."""
        rows = [{
            "strike": s.strike, "dte": s.expiry_days,
            "oi": s.open_interest, "volume": s.daily_volume,
            "spread": s.bid_ask_spread, "rel_spread": s.relative_spread,
            "mid": s.midprice,
        } for s in chain]
        return pd.DataFrame(rows)

    @staticmethod
    def best_strikes(
        chain: List[StrikeLiquidity], dte: int, top_n: int = 5,
    ) -> List[StrikeLiquidity]:
        """Return the most liquid strikes for a given DTE."""
        candidates = [s for s in chain if s.expiry_days == dte]
        candidates.sort(key=lambda s: s.daily_volume, reverse=True)
        return candidates[:top_n]

    # ------------------------------------------------------------------
    # 2. Bid-ask spread monitoring
    # ------------------------------------------------------------------

    @staticmethod
    def spread_surface(chain: List[StrikeLiquidity]) -> pd.DataFrame:
        """Pivot spread data into strike × DTE grid."""
        df = pd.DataFrame([{
            "strike": s.strike, "dte": s.expiry_days, "spread": s.bid_ask_spread,
        } for s in chain])
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="strike", columns="dte", values="spread")

    # ------------------------------------------------------------------
    # 3. Market impact estimation
    # ------------------------------------------------------------------

    def estimate_impact(
        self, contracts: int, strike_liq: StrikeLiquidity,
    ) -> ImpactEstimate:
        """Estimate total execution cost for a given order size.

        Uses square-root impact model: impact ∝ sqrt(participation).
        """
        volume = max(strike_liq.daily_volume, 1)
        participation = contracts / volume

        # Spread cost: half the bid-ask per leg, 2 legs per spread
        spread_cost_per = strike_liq.bid_ask_spread * CONTRACT_MULT

        # Impact: sqrt model calibrated to options markets
        sigma = strike_liq.relative_spread  # use relative spread as vol proxy
        impact_frac = sigma * math.sqrt(max(participation, 0)) * 0.5
        impact_bps = impact_frac * 10000

        total = contracts * (spread_cost_per + impact_frac * strike_liq.midprice * CONTRACT_MULT)

        return ImpactEstimate(
            contracts=contracts, participation_rate=participation,
            spread_cost_per=spread_cost_per, impact_bps=impact_bps,
            total_cost=total,
        )

    # ------------------------------------------------------------------
    # 4. Capacity calculation
    # ------------------------------------------------------------------

    def calculate_capacity(
        self, strike_liq: StrikeLiquidity,
    ) -> CapacityResult:
        """Max contracts before slippage exceeds threshold."""
        max_by_participation = int(strike_liq.daily_volume * self.max_participation)
        max_by_oi = int(strike_liq.open_interest * 0.01)  # 1% of OI

        # Find max where impact < threshold
        max_by_impact = max_by_participation
        for n in range(1, max_by_participation + 1):
            imp = self.estimate_impact(n, strike_liq)
            slip_pct = imp.total_cost / (n * strike_liq.midprice * CONTRACT_MULT) if strike_liq.midprice > 0 else 1
            if slip_pct > self.max_slippage_pct:
                max_by_impact = max(1, n - 1)
                break

        max_contracts = min(max_by_participation, max_by_oi, max_by_impact)
        limiting = "volume"
        if max_contracts == max_by_oi:
            limiting = "oi"
        elif max_contracts == max_by_impact:
            limiting = "impact"

        # Slippage at max
        if max_contracts > 0:
            imp = self.estimate_impact(max_contracts, strike_liq)
            slip = imp.total_cost / (max_contracts * strike_liq.midprice * CONTRACT_MULT) if strike_liq.midprice > 0 else 0
        else:
            slip = 0

        return CapacityResult(
            max_contracts=max(1, max_contracts),
            max_notional=max(1, max_contracts) * strike_liq.midprice * CONTRACT_MULT,
            slippage_at_max=slip,
            limiting_factor=limiting,
        )

    # ------------------------------------------------------------------
    # 5. Adaptive sizing
    # ------------------------------------------------------------------

    def liquidity_score(self, strike_liq: StrikeLiquidity) -> LiquidityScore:
        """Score overall liquidity of a strike (0-1)."""
        # Volume score: 0-1 based on daily volume (10K = 1.0)
        vol_score = min(strike_liq.daily_volume / 10000, 1.0)
        # OI score
        oi_score = min(strike_liq.open_interest / 50000, 1.0)
        # Spread score: tighter = better (invert, $0.02 = 1.0, $0.20 = 0.1)
        spread_score = max(0.1, min(1.0, 0.02 / max(strike_liq.bid_ask_spread, 0.001)))

        composite = vol_score * 0.35 + oi_score * 0.30 + spread_score * 0.35
        constraint = "none"
        if vol_score < 0.3:
            constraint = "volume"
        elif oi_score < 0.3:
            constraint = "oi"
        elif spread_score < 0.3:
            constraint = "spread"

        return LiquidityScore(
            score=composite, avg_spread=strike_liq.bid_ask_spread,
            avg_oi=float(strike_liq.open_interest),
            avg_volume=float(strike_liq.daily_volume),
            constraint=constraint,
        )

    def adaptive_size(
        self, base_contracts: int, strike_liq: StrikeLiquidity,
    ) -> AdaptiveSize:
        """Scale position size by liquidity conditions."""
        lscore = self.liquidity_score(strike_liq)
        cap = self.calculate_capacity(strike_liq)

        # Scale factor: full size at score >= 0.7, half at 0.3, zero below 0.1
        if lscore.score >= 0.7:
            scale = 1.0
        elif lscore.score >= 0.3:
            scale = 0.3 + (lscore.score - 0.3) / 0.4 * 0.7
        else:
            scale = max(0.1, lscore.score / 0.3 * 0.3)

        adjusted = max(1, min(int(base_contracts * scale), cap.max_contracts))
        reason = "full" if scale >= 0.95 else (
            f"scaled {scale:.0%} ({lscore.constraint})" if lscore.constraint != "none"
            else f"scaled {scale:.0%}")

        return AdaptiveSize(
            base_contracts=base_contracts, adjusted_contracts=adjusted,
            liquidity_score=lscore.score, scale_factor=scale, reason=reason,
        )

    # ------------------------------------------------------------------
    # 6. Roll optimisation
    # ------------------------------------------------------------------

    def optimal_roll(
        self,
        current_strike: float, current_dte: int,
        chain: List[StrikeLiquidity],
        target_dte_range: Tuple[int, int] = (25, 45),
    ) -> List[RollPath]:
        """Find cheapest roll path considering liquidity."""
        candidates = [
            s for s in chain
            if target_dte_range[0] <= s.expiry_days <= target_dte_range[1]
            and s.open_interest >= self.min_oi
        ]

        if not candidates:
            return []

        paths: List[RollPath] = []
        for target in candidates:
            # Roll cost: close current (spread cost) + open target (spread cost)
            # Approximate: each leg costs half the bid-ask
            current_chain = [s for s in chain if s.strike == current_strike
                              and s.expiry_days == current_dte]
            close_cost = current_chain[0].bid_ask_spread if current_chain else 0.05
            open_cost = target.bid_ask_spread

            roll_cost = (close_cost + open_cost) * CONTRACT_MULT
            lscore = self.liquidity_score(target)

            paths.append(RollPath(
                current_strike=current_strike, current_dte=current_dte,
                target_strike=target.strike, target_dte=target.expiry_days,
                roll_cost=roll_cost, liquidity_score=lscore.score,
                is_optimal=False,
            ))

        # Score: minimise cost while maximising liquidity
        for p in paths:
            p._score = -p.roll_cost + p.liquidity_score * 5  # type: ignore

        paths.sort(key=lambda p: getattr(p, '_score', 0), reverse=True)
        if paths:
            paths[0].is_optimal = True

        return paths[:5]

    # ------------------------------------------------------------------
    # 7. Backtest comparison
    # ------------------------------------------------------------------

    def backtest(
        self,
        returns: pd.Series,
        chain_per_day: Optional[Dict[int, List[StrikeLiquidity]]] = None,
        base_contracts: int = 5,
        vix_series: Optional[pd.Series] = None,
        seed: int = 42,
    ) -> SizingBacktestResult:
        """Compare fixed vs liquidity-aware sizing over return series."""
        n = len(returns)
        rng = np.random.default_rng(seed)

        fixed_rets: List[float] = []
        adaptive_rets: List[float] = []
        fixed_slip_total = 0.0
        adaptive_slip_total = 0.0
        n_trades = 0

        for i in range(n):
            r = float(returns.iloc[i])

            # Generate daily chain (or use provided)
            vix = float(vix_series.iloc[i]) if vix_series is not None else 20 + rng.normal(0, 3)
            vix = max(12, min(60, vix))

            if chain_per_day and i in chain_per_day:
                chain = chain_per_day[i]
            else:
                chain = generate_option_chain(underlying=450, vix=vix, seed=seed + i)

            # Pick ATM strike
            atm = [s for s in chain if abs(s.strike - 450) < 3 and s.expiry_days == 30]
            if not atm:
                fixed_rets.append(r * base_contracts * 0.01)
                adaptive_rets.append(r * base_contracts * 0.01)
                continue

            strike_liq = atm[0]
            n_trades += 1

            # Fixed sizing
            fixed_imp = self.estimate_impact(base_contracts, strike_liq)
            fixed_slip = fixed_imp.total_cost / max(base_contracts * strike_liq.midprice * CONTRACT_MULT, 1)
            fixed_slip_total += fixed_imp.total_cost
            fixed_rets.append(r * base_contracts * 0.01 - fixed_slip / TRADING_DAYS)

            # Adaptive sizing
            adapted = self.adaptive_size(base_contracts, strike_liq)
            adap_imp = self.estimate_impact(adapted.adjusted_contracts, strike_liq)
            adap_slip = adap_imp.total_cost / max(adapted.adjusted_contracts * strike_liq.midprice * CONTRACT_MULT, 1)
            adaptive_slip_total += adap_imp.total_cost
            adaptive_rets.append(r * adapted.adjusted_contracts * 0.01 - adap_slip / TRADING_DAYS)

        def _metrics(rets_list):
            arr = np.array(rets_list)
            total = float((1 + arr).prod() - 1)
            n_yr = len(arr) / TRADING_DAYS
            annual = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
            mu = float(arr.mean())
            std = float(arr.std())
            sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
            eq = np.cumprod(1 + arr)
            dd = float((1 - eq / np.maximum.accumulate(eq)).max())
            return annual, sharpe, dd

        f_ret, f_sh, f_dd = _metrics(fixed_rets)
        a_ret, a_sh, a_dd = _metrics(adaptive_rets)
        slip_reduction = (fixed_slip_total - adaptive_slip_total) / max(fixed_slip_total, 1)

        return SizingBacktestResult(
            fixed_return=f_ret, fixed_sharpe=f_sh, fixed_max_dd=f_dd,
            fixed_total_slippage=fixed_slip_total,
            adaptive_return=a_ret, adaptive_sharpe=a_sh, adaptive_max_dd=a_dd,
            adaptive_total_slippage=adaptive_slip_total,
            slippage_reduction=slip_reduction, n_trades=n_trades,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: SizingBacktestResult,
        chain: Optional[List[StrikeLiquidity]] = None,
        output_path: str = "reports/liquidity_sizer.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Liquidity-Aware Sizing</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; font-weight: 500; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
.green {{ color: #059669; font-weight: 700; }}
</style></head><body>
<h1>EXP-1200-max: Liquidity-Aware Position Sizing</h1>
<div class="card">
<p><strong>Slippage Reduction:</strong> <span class="green">{r.slippage_reduction:.1%}</span> |
<strong>Trades:</strong> {r.n_trades}</p>
</div>

<h2>Fixed vs Adaptive Sizing</h2>
<table>
<tr><th>Metric</th><th>Fixed Sizing</th><th>Adaptive Sizing</th><th>Improvement</th></tr>
<tr><td>Annual Return</td><td>{r.fixed_return:.2%}</td><td>{r.adaptive_return:.2%}</td>
<td>{r.adaptive_return - r.fixed_return:+.2%}</td></tr>
<tr><td>Sharpe</td><td>{r.fixed_sharpe:.2f}</td><td>{r.adaptive_sharpe:.2f}</td>
<td>{r.adaptive_sharpe - r.fixed_sharpe:+.2f}</td></tr>
<tr><td>Max Drawdown</td><td>{r.fixed_max_dd:.2%}</td><td>{r.adaptive_max_dd:.2%}</td>
<td>{r.fixed_max_dd - r.adaptive_max_dd:+.2%}</td></tr>
<tr><td>Total Slippage</td><td>${r.fixed_total_slippage:,.0f}</td>
<td>${r.adaptive_total_slippage:,.0f}</td>
<td class="green">-${r.fixed_total_slippage - r.adaptive_total_slippage:,.0f}</td></tr>
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
