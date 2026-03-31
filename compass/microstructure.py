"""
Market microstructure analysis engine.

Combines academic microstructure models into a single analysis toolkit
for credit-spread execution quality:

  1. Bid-ask spread estimation  (Roll model, effective spread, realised spread)
  2. Order flow imbalance       (VPIN, Kyle lambda, signed volume flow)
  3. Price impact estimation    (temporary + permanent decomposition)
  4. Trade classification       (Lee-Ready tick test)
  5. Intraday volatility        (U-shape detection, overnight gaps)
  6. Informed trading probability (simplified PIN)
  7. Liquidity measurement      (Amihud illiquidity, turnover ratio)
  8. HTML report

All methods operate on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpreadEstimate:
    """Bid-ask spread estimates."""
    roll_spread: float = 0.0
    effective_spread: float = 0.0
    realised_spread: float = 0.0
    quoted_spread: float = 0.0
    relative_spread: float = 0.0


@dataclass
class OrderFlowMetrics:
    """Order flow imbalance and toxicity."""
    vpin: float = 0.0
    kyle_lambda: float = 0.0
    kyle_r_squared: float = 0.0
    net_order_flow: float = 0.0
    buy_volume_pct: float = 0.5
    imbalance_ratio: float = 0.0


@dataclass
class PriceImpactEstimate:
    """Price impact decomposition."""
    total_impact: float = 0.0
    permanent_impact: float = 0.0
    temporary_impact: float = 0.0
    r_squared: float = 0.0


@dataclass
class TradeClassification:
    """Lee-Ready tick test classification of a trade."""
    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    n_unclassified: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_pct: float = 0.5


@dataclass
class IntradayVolatility:
    """Intraday volatility bucket."""
    bucket: str
    volatility: float
    avg_volume: float
    avg_spread: float
    n_obs: int


@dataclass
class OvernightGap:
    """Overnight gap measurement."""
    date: datetime
    gap_return: float
    abs_gap: float


@dataclass
class InformedTradingEstimate:
    """Simplified PIN (probability of informed trading)."""
    pin: float = 0.0
    alpha: float = 0.0       # prob of information event
    delta: float = 0.5       # prob bad news given event
    mu: float = 0.0          # informed arrival rate
    epsilon_b: float = 0.0   # uninformed buy rate
    epsilon_s: float = 0.0   # uninformed sell rate


@dataclass
class LiquidityMetrics:
    """Liquidity measurement."""
    amihud_illiquidity: float = 0.0
    turnover_ratio: float = 0.0
    avg_daily_volume: float = 0.0
    avg_dollar_volume: float = 0.0
    zero_return_days_pct: float = 0.0


@dataclass
class MicrostructureSummary:
    """Full microstructure snapshot."""
    spreads: SpreadEstimate
    order_flow: OrderFlowMetrics
    price_impact: PriceImpactEstimate
    trade_class: TradeClassification
    informed_trading: InformedTradingEstimate
    liquidity: LiquidityMetrics
    intraday_patterns: List[IntradayVolatility] = field(default_factory=list)
    overnight_gaps: List[OvernightGap] = field(default_factory=list)
    u_shape_detected: bool = False


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class MicrostructureEngine:
    """Market microstructure analysis engine.

    Args:
        n_vpin_buckets: Volume buckets for VPIN computation.
        impact_lag: Lag for permanent/temporary impact decomposition.
    """

    def __init__(
        self,
        n_vpin_buckets: int = 50,
        impact_lag: int = 5,
    ) -> None:
        self.n_vpin_buckets = n_vpin_buckets
        self.impact_lag = impact_lag

    # ------------------------------------------------------------------
    # 1. Spread estimation
    # ------------------------------------------------------------------

    @staticmethod
    def roll_spread(prices: pd.Series) -> float:
        """Roll (1984) spread estimator: 2 * sqrt(-Cov(dp_t, dp_{t-1}))."""
        dp = prices.diff().dropna()
        if len(dp) < 3:
            return 0.0
        cov = float(dp.autocorr(lag=1) * dp.var())
        return 2.0 * np.sqrt(-cov) if cov < 0 else 0.0

    @staticmethod
    def effective_spread(
        trade_prices: pd.Series, mid_prices: pd.Series,
    ) -> float:
        """Effective spread = 2 * mean(|trade - mid|)."""
        diff = (trade_prices - mid_prices).dropna()
        if diff.empty:
            return 0.0
        return float(2.0 * diff.abs().mean())

    @staticmethod
    def realised_spread(
        trade_prices: pd.Series, mid_prices: pd.Series,
        direction: pd.Series, lag: int = 5,
    ) -> float:
        """Realised spread: 2 * d_t * (p_t - m_{t+lag}) / p_t averaged.

        direction: +1 for buys, -1 for sells.
        """
        future_mid = mid_prices.shift(-lag)
        aligned = pd.DataFrame({
            "tp": trade_prices, "fm": future_mid, "d": direction,
        }).dropna()
        if aligned.empty:
            return 0.0
        rs = 2.0 * aligned["d"] * (aligned["tp"] - aligned["fm"]) / aligned["tp"]
        return float(rs.mean())

    @staticmethod
    def quoted_spread(bid: pd.Series, ask: pd.Series) -> float:
        s = (ask - bid).dropna()
        return float(s.mean()) if not s.empty else 0.0

    @staticmethod
    def relative_spread(bid: pd.Series, ask: pd.Series) -> float:
        mid = (bid + ask) / 2.0
        rel = ((ask - bid) / mid).dropna()
        return float(rel.mean()) if not rel.empty else 0.0

    def estimate_spreads(
        self,
        prices: pd.Series,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
        trade_prices: Optional[pd.Series] = None,
        mid_prices: Optional[pd.Series] = None,
        direction: Optional[pd.Series] = None,
    ) -> SpreadEstimate:
        """Compute all available spread metrics."""
        roll = self.roll_spread(prices)
        eff = self.effective_spread(trade_prices, mid_prices) if trade_prices is not None and mid_prices is not None else 0.0
        real = 0.0
        if trade_prices is not None and mid_prices is not None and direction is not None:
            real = self.realised_spread(trade_prices, mid_prices, direction)
        quot = self.quoted_spread(bid, ask) if bid is not None and ask is not None else 0.0
        rel = self.relative_spread(bid, ask) if bid is not None and ask is not None else 0.0
        return SpreadEstimate(
            roll_spread=roll, effective_spread=eff, realised_spread=real,
            quoted_spread=quot, relative_spread=rel,
        )

    def rolling_roll_spread(
        self, prices: pd.Series, window: int = 21,
    ) -> pd.Series:
        """Rolling Roll spread."""
        return prices.rolling(window).apply(
            lambda x: self.roll_spread(x), raw=False,
        )

    # ------------------------------------------------------------------
    # 2. Order flow imbalance
    # ------------------------------------------------------------------

    def compute_vpin(
        self, volume: pd.Series, price_changes: pd.Series,
    ) -> float:
        """Volume-synchronised Probability of Informed Trading."""
        aligned = pd.DataFrame({"vol": volume, "dp": price_changes}).dropna()
        if aligned.empty or aligned["vol"].sum() <= 0:
            return 0.0

        buy_vol = aligned["vol"].where(aligned["dp"] > 0, 0.0)
        sell_vol = aligned["vol"].where(aligned["dp"] <= 0, 0.0)
        total = aligned["vol"].sum()
        bucket_size = total / max(self.n_vpin_buckets, 1)
        if bucket_size <= 0:
            return 0.0

        cum_vol = cum_buy = cum_sell = 0.0
        imbalances: List[float] = []

        for i in range(len(aligned)):
            bv = float(buy_vol.iloc[i])
            sv = float(sell_vol.iloc[i])
            v = float(aligned["vol"].iloc[i])
            cum_buy += bv
            cum_sell += sv
            cum_vol += v

            while cum_vol >= bucket_size and len(imbalances) < self.n_vpin_buckets:
                imbalances.append(abs(cum_buy - cum_sell) / bucket_size)
                cum_vol -= bucket_size
                ratio = min(cum_vol / v, 1.0) if v > 0 else 0.0
                cum_buy = bv * ratio
                cum_sell = sv * ratio

        return float(np.mean(imbalances)) if imbalances else 0.0

    @staticmethod
    def kyle_lambda(
        price_changes: pd.Series, signed_volume: pd.Series,
    ) -> Tuple[float, float]:
        """Kyle lambda: dp = lambda * signed_flow + eps.  Returns (lambda, r2)."""
        aligned = pd.DataFrame({"dp": price_changes, "sv": signed_volume}).dropna()
        if len(aligned) < 5:
            return 0.0, 0.0
        y = aligned["dp"].values
        x = np.column_stack([np.ones(len(y)), aligned["sv"].values])
        try:
            b, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        pred = x @ b
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(b[1]), r2

    def compute_order_flow(
        self,
        volume: pd.Series,
        price_changes: pd.Series,
        signed_volume: Optional[pd.Series] = None,
    ) -> OrderFlowMetrics:
        """Full order flow metrics."""
        vpin = self.compute_vpin(volume, price_changes)
        kl, kr = (0.0, 0.0)
        if signed_volume is not None:
            kl, kr = self.kyle_lambda(price_changes, signed_volume)

        buy_vol = volume.where(price_changes > 0, 0.0).sum()
        sell_vol = volume.where(price_changes <= 0, 0.0).sum()
        total = buy_vol + sell_vol
        buy_pct = buy_vol / total if total > 0 else 0.5
        imbalance = abs(buy_vol - sell_vol) / total if total > 0 else 0.0

        return OrderFlowMetrics(
            vpin=vpin, kyle_lambda=kl, kyle_r_squared=kr,
            net_order_flow=float(buy_vol - sell_vol),
            buy_volume_pct=float(buy_pct),
            imbalance_ratio=float(imbalance),
        )

    # ------------------------------------------------------------------
    # 3. Price impact estimation
    # ------------------------------------------------------------------

    def estimate_price_impact(
        self,
        price_changes: pd.Series,
        signed_volume: pd.Series,
    ) -> PriceImpactEstimate:
        """Decompose price impact into permanent and temporary."""
        aligned = pd.DataFrame({"dp": price_changes, "sv": signed_volume}).dropna()
        if len(aligned) < self.impact_lag + 5:
            return PriceImpactEstimate()

        y = aligned["dp"].values
        x_c = np.column_stack([np.ones(len(y)), aligned["sv"].values])
        try:
            b_imm, _, _, _ = np.linalg.lstsq(x_c, y, rcond=None)
        except np.linalg.LinAlgError:
            return PriceImpactEstimate()
        total_lam = float(b_imm[1])

        pred = x_c @ b_imm
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        cum_dp = aligned["dp"].rolling(self.impact_lag).sum().shift(-self.impact_lag + 1).dropna()
        sv_a = aligned["sv"].iloc[:len(cum_dp)]
        if len(cum_dp) < 5:
            return PriceImpactEstimate(total_impact=total_lam, r_squared=r2)

        x_p = np.column_stack([np.ones(len(cum_dp)), sv_a.values])
        try:
            b_perm, _, _, _ = np.linalg.lstsq(x_p, cum_dp.values, rcond=None)
        except np.linalg.LinAlgError:
            return PriceImpactEstimate(total_impact=total_lam, r_squared=r2)

        perm = float(b_perm[1])
        temp = total_lam - perm
        return PriceImpactEstimate(
            total_impact=total_lam, permanent_impact=perm,
            temporary_impact=temp, r_squared=r2,
        )

    # ------------------------------------------------------------------
    # 4. Trade classification — Lee-Ready tick test
    # ------------------------------------------------------------------

    @staticmethod
    def lee_ready_classify(
        trade_prices: pd.Series,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> TradeClassification:
        """Lee-Ready (1991) trade classification.

        Step 1: Quote test — if trade > mid → buy; < mid → sell.
        Step 2: Tick test fallback — if trade > prev_trade → buy; etc.
        """
        if trade_prices.empty:
            return TradeClassification()

        n = len(trade_prices)
        signs = np.zeros(n, dtype=int)

        # Quote test if bid/ask available
        has_quotes = bid is not None and ask is not None
        if has_quotes:
            mid = (bid + ask) / 2.0
            aligned = pd.DataFrame({
                "tp": trade_prices, "mid": mid,
            }).reindex(trade_prices.index)
            for i in range(n):
                if pd.notna(aligned["mid"].iloc[i]):
                    diff = aligned["tp"].iloc[i] - aligned["mid"].iloc[i]
                    if diff > 1e-10:
                        signs[i] = 1
                    elif diff < -1e-10:
                        signs[i] = -1

        # Tick test for unclassified
        tp_vals = trade_prices.values
        for i in range(n):
            if signs[i] == 0 and i > 0:
                tick = tp_vals[i] - tp_vals[i - 1]
                if tick > 1e-10:
                    signs[i] = 1
                elif tick < -1e-10:
                    signs[i] = -1

        vol = volume.values if volume is not None else np.ones(n)
        n_buys = int((signs == 1).sum())
        n_sells = int((signs == -1).sum())
        n_unc = int((signs == 0).sum())
        buy_vol = float(vol[signs == 1].sum()) if n_buys > 0 else 0.0
        sell_vol = float(vol[signs == -1].sum()) if n_sells > 0 else 0.0
        total = buy_vol + sell_vol
        buy_pct = buy_vol / total if total > 0 else 0.5

        return TradeClassification(
            n_trades=n, n_buys=n_buys, n_sells=n_sells,
            n_unclassified=n_unc, buy_volume=buy_vol,
            sell_volume=sell_vol, buy_pct=buy_pct,
        )

    # ------------------------------------------------------------------
    # 5. Intraday volatility
    # ------------------------------------------------------------------

    @staticmethod
    def intraday_volatility(
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        spreads: Optional[pd.Series] = None,
    ) -> List[IntradayVolatility]:
        """Compute volatility/volume/spread by hour of day."""
        if not hasattr(returns.index, "hour"):
            return []
        df = pd.DataFrame({"ret": returns, "hour": returns.index.hour})
        if volume is not None:
            df["vol"] = volume.reindex(returns.index).fillna(0)
        else:
            df["vol"] = 0.0
        if spreads is not None:
            df["spr"] = spreads.reindex(returns.index).fillna(0)
        else:
            df["spr"] = 0.0

        result: List[IntradayVolatility] = []
        for hour, grp in df.groupby("hour"):
            result.append(IntradayVolatility(
                bucket=f"{int(hour):02d}:00",
                volatility=float(grp["ret"].std() * np.sqrt(TRADING_DAYS)) if len(grp) > 1 else 0.0,
                avg_volume=float(grp["vol"].mean()),
                avg_spread=float(grp["spr"].mean()),
                n_obs=len(grp),
            ))
        return result

    @staticmethod
    def detect_u_shape(patterns: List[IntradayVolatility]) -> bool:
        """Detect U-shaped intraday vol (higher at open/close than mid-day)."""
        if len(patterns) < 3:
            return False
        vols = [p.volatility for p in patterns]
        n = len(vols)
        edge_avg = (vols[0] + vols[-1]) / 2.0
        mid_s = n // 4
        mid_e = 3 * n // 4
        mid_avg = float(np.mean(vols[mid_s:mid_e])) if mid_e > mid_s else 0.0
        return edge_avg > mid_avg * 1.05

    @staticmethod
    def overnight_gaps(
        open_prices: pd.Series, close_prices: pd.Series,
    ) -> List[OvernightGap]:
        """Compute overnight gap returns: open_t / close_{t-1} - 1."""
        prev_close = close_prices.shift(1)
        gap = (open_prices / prev_close - 1).dropna()
        return [
            OvernightGap(date=dt, gap_return=float(g), abs_gap=abs(float(g)))
            for dt, g in gap.items()
        ]

    # ------------------------------------------------------------------
    # 6. Informed trading probability (simplified PIN)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_pin(
        buy_counts: pd.Series,
        sell_counts: pd.Series,
    ) -> InformedTradingEstimate:
        """Simplified PIN estimation using method-of-moments.

        PIN = alpha * mu / (alpha * mu + epsilon_b + epsilon_s)
        where:
          epsilon_b = avg daily buys (uninformed)
          epsilon_s = avg daily sells (uninformed)
          mu = excess arrival rate on information days
          alpha = fraction of days with information events
        """
        b = buy_counts.values.astype(float)
        s = sell_counts.values.astype(float)
        if len(b) < 5:
            return InformedTradingEstimate()

        avg_b = float(b.mean())
        avg_s = float(s.mean())

        # Days with big imbalance are "information days"
        imb = np.abs(b - s)
        threshold = float(np.percentile(imb, 75))
        info_days = imb > threshold
        alpha = float(info_days.mean())

        if alpha < 0.01:
            return InformedTradingEstimate(
                pin=0.0, alpha=0.0, delta=0.5,
                mu=0.0, epsilon_b=avg_b, epsilon_s=avg_s,
            )

        # On information days, which side has excess?
        info_buys = float(b[info_days].mean())
        info_sells = float(s[info_days].mean())
        delta = 0.5
        if info_buys + info_sells > 0:
            delta = info_sells / (info_buys + info_sells)

        mu = float(imb[info_days].mean())
        eps_b = float(b[~info_days].mean()) if (~info_days).sum() > 0 else avg_b
        eps_s = float(s[~info_days].mean()) if (~info_days).sum() > 0 else avg_s

        denom = alpha * mu + eps_b + eps_s
        pin = alpha * mu / denom if denom > 0 else 0.0

        return InformedTradingEstimate(
            pin=pin, alpha=alpha, delta=delta,
            mu=mu, epsilon_b=eps_b, epsilon_s=eps_s,
        )

    # ------------------------------------------------------------------
    # 7. Liquidity measurement
    # ------------------------------------------------------------------

    @staticmethod
    def amihud_illiquidity(
        returns: pd.Series,
        dollar_volume: pd.Series,
    ) -> float:
        """Amihud (2002) illiquidity: mean(|r_t| / DolVol_t)."""
        aligned = pd.DataFrame({
            "absret": returns.abs(), "dvol": dollar_volume,
        }).dropna()
        aligned = aligned[aligned["dvol"] > 0]
        if aligned.empty:
            return 0.0
        return float((aligned["absret"] / aligned["dvol"]).mean())

    @staticmethod
    def turnover_ratio(
        volume: pd.Series,
        shares_outstanding: float,
    ) -> float:
        """Average daily turnover ratio."""
        if shares_outstanding <= 0 or volume.empty:
            return 0.0
        return float(volume.mean() / shares_outstanding)

    def compute_liquidity(
        self,
        returns: pd.Series,
        volume: pd.Series,
        prices: pd.Series,
        shares_outstanding: float = 0.0,
    ) -> LiquidityMetrics:
        """Full liquidity metrics."""
        dvol = volume * prices
        amihud = self.amihud_illiquidity(returns, dvol)
        tr = self.turnover_ratio(volume, shares_outstanding) if shares_outstanding > 0 else 0.0
        avg_vol = float(volume.mean())
        avg_dvol = float(dvol.mean())
        zero_pct = float((returns.abs() < 1e-10).sum() / len(returns)) if len(returns) > 0 else 0.0
        return LiquidityMetrics(
            amihud_illiquidity=amihud, turnover_ratio=tr,
            avg_daily_volume=avg_vol, avg_dollar_volume=avg_dvol,
            zero_return_days_pct=zero_pct,
        )

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        prices: pd.Series,
        volume: pd.Series,
        returns: Optional[pd.Series] = None,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
        open_prices: Optional[pd.Series] = None,
        close_prices: Optional[pd.Series] = None,
        buy_counts: Optional[pd.Series] = None,
        sell_counts: Optional[pd.Series] = None,
        shares_outstanding: float = 0.0,
    ) -> MicrostructureSummary:
        """Run all analyses and return a full summary."""
        if returns is None:
            returns = prices.pct_change().dropna()

        dp = prices.diff().dropna()

        # Direction from tick test
        direction = pd.Series(
            np.sign(dp.values), index=dp.index,
        ).replace(0, np.nan).ffill().fillna(1)

        signed_vol = direction.reindex(volume.index).fillna(0) * volume

        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0

        spreads = self.estimate_spreads(
            prices, bid=bid, ask=ask,
            trade_prices=prices if mid is not None else None,
            mid_prices=mid, direction=direction,
        )
        order_flow = self.compute_order_flow(volume, dp, signed_vol)
        impact = self.estimate_price_impact(dp, signed_vol)
        trade_class = self.lee_ready_classify(
            prices, bid=bid, ask=ask, volume=volume,
        )

        # Intraday patterns
        intraday = []
        if hasattr(returns.index, "hour"):
            intraday = self.intraday_volatility(returns, volume=volume)
        u_shape = self.detect_u_shape(intraday)

        gaps: List[OvernightGap] = []
        if open_prices is not None and close_prices is not None:
            gaps = self.overnight_gaps(open_prices, close_prices)

        pin_est = InformedTradingEstimate()
        if buy_counts is not None and sell_counts is not None:
            pin_est = self.estimate_pin(buy_counts, sell_counts)

        liquidity = self.compute_liquidity(
            returns, volume, prices, shares_outstanding,
        )

        return MicrostructureSummary(
            spreads=spreads, order_flow=order_flow,
            price_impact=impact, trade_class=trade_class,
            informed_trading=pin_est, liquidity=liquidity,
            intraday_patterns=intraday, overnight_gaps=gaps,
            u_shape_detected=u_shape,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 700, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        vmin = min(values)
        vmax = max(values)
        if vmax <= vmin:
            vmax = vmin + 0.01
        pad = 50
        pw = width - 2 * pad
        ph = height - 55

        def tx(i: int) -> float:
            return pad + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return 28 + (1 - (v - vmin) / (vmax - vmin)) * ph

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                      for i, v in enumerate(values))
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_bar(
        labels: List[str], values: List[float], title: str,
        width: int = 600, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if not values:
            return ""
        n = len(values)
        vmax = max(max(values), 0.001)
        pad_l, pad_b = 60, 40
        pw = width - pad_l - 20
        ph = height - 50 - pad_b
        bw = pw / max(n, 1) * 0.7
        gap = pw / max(n, 1)

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        for i in range(n):
            x = pad_l + i * gap + (gap - bw) / 2
            bh = values[i] / vmax * ph
            y = 30 + ph - bh
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{bh:.0f}" '
                     f'fill="{color}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{height - 10:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#666">{labels[i]}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        summary: MicrostructureSummary,
        rolling_spreads: Optional[List[float]] = None,
        output_path: str = "reports/microstructure.html",
    ) -> str:
        """HTML report: spread analysis, order flow, liquidity dashboard."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        s = summary.spreads
        of = summary.order_flow
        pi = summary.price_impact
        tc = summary.trade_class
        it = summary.informed_trading
        lq = summary.liquidity

        # Charts
        spread_svg = ""
        if rolling_spreads and len(rolling_spreads) > 2:
            spread_svg = self._svg_line(rolling_spreads, "Rolling Spread", color="#2980b9")

        intraday_svg = ""
        if summary.intraday_patterns:
            labs = [p.bucket for p in summary.intraday_patterns]
            vals = [p.volatility for p in summary.intraday_patterns]
            intraday_svg = self._svg_bar(labs, vals, "Intraday Volatility", color="#e67e22")

        # Gap chart
        gap_svg = ""
        if summary.overnight_gaps and len(summary.overnight_gaps) > 2:
            gvals = [g.gap_return for g in summary.overnight_gaps]
            gap_svg = self._svg_line(gvals, "Overnight Gaps", color="#8e44ad")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Microstructure Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.m {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
</style></head><body>
<h1>Market Microstructure Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>U-Shape Detected:</strong> {'Yes' if summary.u_shape_detected else 'No'} |
   <strong>Overnight Gaps:</strong> {len(summary.overnight_gaps)}</p>
</div>

<h2>Spread Analysis</h2>
{spread_svg}
<table class="m"><tr><th>Roll</th><th>Effective</th><th>Realised</th><th>Quoted</th><th>Relative</th></tr>
<tr><td>{s.roll_spread:.6f}</td><td>{s.effective_spread:.6f}</td><td>{s.realised_spread:.6f}</td>
<td>{s.quoted_spread:.6f}</td><td>{s.relative_spread:.4%}</td></tr></table>

<h2>Order Flow</h2>
<table class="m"><tr><th>VPIN</th><th>Kyle &lambda;</th><th>R&sup2;</th>
<th>Buy Vol %</th><th>Imbalance</th></tr>
<tr><td>{of.vpin:.4f}</td><td>{of.kyle_lambda:.6f}</td><td>{of.kyle_r_squared:.4f}</td>
<td>{of.buy_volume_pct:.1%}</td><td>{of.imbalance_ratio:.4f}</td></tr></table>

<h2>Price Impact</h2>
<table class="m"><tr><th>Total</th><th>Permanent</th><th>Temporary</th><th>R&sup2;</th></tr>
<tr><td>{pi.total_impact:.6f}</td><td>{pi.permanent_impact:.6f}</td>
<td>{pi.temporary_impact:.6f}</td><td>{pi.r_squared:.4f}</td></tr></table>

<h2>Trade Classification (Lee-Ready)</h2>
<table class="m"><tr><th>Trades</th><th>Buys</th><th>Sells</th><th>Unclassified</th><th>Buy %</th></tr>
<tr><td>{tc.n_trades}</td><td>{tc.n_buys}</td><td>{tc.n_sells}</td>
<td>{tc.n_unclassified}</td><td>{tc.buy_pct:.1%}</td></tr></table>

<h2>Informed Trading (PIN)</h2>
<table class="m"><tr><th>PIN</th><th>&alpha;</th><th>&delta;</th><th>&mu;</th>
<th>&epsilon;<sub>B</sub></th><th>&epsilon;<sub>S</sub></th></tr>
<tr><td>{it.pin:.4f}</td><td>{it.alpha:.4f}</td><td>{it.delta:.4f}</td>
<td>{it.mu:.2f}</td><td>{it.epsilon_b:.2f}</td><td>{it.epsilon_s:.2f}</td></tr></table>

<h2>Liquidity Dashboard</h2>
<table class="m"><tr><th>Amihud</th><th>Turnover</th><th>Avg Volume</th>
<th>Avg $ Volume</th><th>Zero-Return %</th></tr>
<tr><td>{lq.amihud_illiquidity:.2e}</td><td>{lq.turnover_ratio:.4%}</td>
<td>{lq.avg_daily_volume:,.0f}</td><td>{lq.avg_dollar_volume:,.0f}</td>
<td>{lq.zero_return_days_pct:.1%}</td></tr></table>

<div class="grid">
<div>{intraday_svg}</div>
<div>{gap_svg}</div>
</div>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Microstructure report -> %s", path)
        return str(path)
