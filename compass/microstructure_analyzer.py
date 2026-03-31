"""
Market microstructure analyzer — bid-ask spreads, price impact,
order flow toxicity, intraday patterns, and execution quality.

Components:
  1. Bid-ask spread estimation  (Roll model, effective spread)
  2. Price impact modelling     (Kyle lambda, permanent vs temporary)
  3. Order flow toxicity        (VPIN, order imbalance)
  4. Intraday volatility        (U-shape detection, overnight gaps)
  5. Optimal execution windows  (lowest spread, highest liquidity)
  6. Maker vs taker analysis    (fill quality by side)
  7. HTML report                (spread, toxicity, heatmap)

All methods work on pre-loaded data — no network calls.
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
    """Bid-ask spread estimate for a period."""
    date: Optional[datetime] = None
    roll_spread: float = 0.0          # Roll (1984) model
    effective_spread: float = 0.0     # 2 * |price - midpoint|
    quoted_spread: float = 0.0        # ask - bid
    relative_spread: float = 0.0      # spread / midprice


@dataclass
class PriceImpact:
    """Price impact decomposition."""
    kyle_lambda: float = 0.0          # permanent impact per unit flow
    permanent_impact: float = 0.0
    temporary_impact: float = 0.0
    total_impact: float = 0.0
    r_squared: float = 0.0


@dataclass
class ToxicityMetrics:
    """Order flow toxicity snapshot."""
    vpin: float = 0.0                 # Volume-synchronised PIN
    order_imbalance: float = 0.0      # |buy_vol - sell_vol| / total_vol
    buy_ratio: float = 0.5
    toxicity_level: str = "normal"    # low / normal / elevated / toxic


@dataclass
class IntradayPattern:
    """Intraday volatility pattern for one bucket."""
    bucket_label: str
    avg_volatility: float
    avg_spread: float
    avg_volume: float
    n_observations: int


@dataclass
class OvernightGap:
    """Overnight gap measurement."""
    date: datetime
    gap_return: float                 # open / prev_close - 1
    abs_gap: float


@dataclass
class ExecutionWindow:
    """Optimal execution window recommendation."""
    bucket_label: str
    avg_spread: float
    avg_volume: float
    quality_score: float              # higher = better for execution
    rank: int


@dataclass
class MakerTakerStats:
    """Maker vs taker fill quality."""
    side: str                         # "maker" or "taker"
    n_fills: int
    avg_price_improvement: float      # vs midpoint
    avg_spread_captured: float
    fill_rate: float


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class MicrostructureAnalyzer:
    """Market microstructure analysis engine.

    Args:
        n_vpin_buckets: Number of volume buckets for VPIN calculation.
        intraday_buckets: Number of time buckets for intraday patterns.
        toxicity_threshold: VPIN above this = 'toxic'.
    """

    def __init__(
        self,
        n_vpin_buckets: int = 50,
        intraday_buckets: int = 13,
        toxicity_threshold: float = 0.7,
    ) -> None:
        self.n_vpin_buckets = n_vpin_buckets
        self.intraday_buckets = intraday_buckets
        self.toxicity_threshold = toxicity_threshold

    # ------------------------------------------------------------------
    # 1. Bid-ask spread estimation
    # ------------------------------------------------------------------

    @staticmethod
    def roll_spread(prices: pd.Series) -> float:
        """Roll (1984) spread estimator from serial covariance.

        spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))  when cov < 0, else 0.
        """
        dp = prices.diff().dropna()
        if len(dp) < 3:
            return 0.0
        cov = float(dp.autocorr(lag=1) * dp.var())
        if cov >= 0:
            return 0.0
        return 2.0 * np.sqrt(-cov)

    @staticmethod
    def effective_spread(
        trade_prices: pd.Series,
        mid_prices: pd.Series,
    ) -> float:
        """Average effective spread = 2 * mean(|trade - mid|)."""
        aligned = pd.DataFrame({"trade": trade_prices, "mid": mid_prices}).dropna()
        if aligned.empty:
            return 0.0
        return float(2.0 * (aligned["trade"] - aligned["mid"]).abs().mean())

    @staticmethod
    def quoted_spread(bid: pd.Series, ask: pd.Series) -> float:
        """Average quoted spread = mean(ask - bid)."""
        spread = (ask - bid).dropna()
        if spread.empty:
            return 0.0
        return float(spread.mean())

    @staticmethod
    def relative_spread(bid: pd.Series, ask: pd.Series) -> float:
        """Average relative spread = mean((ask - bid) / midprice)."""
        mid = (bid + ask) / 2.0
        spread = ((ask - bid) / mid).dropna()
        if spread.empty:
            return 0.0
        return float(spread.mean())

    def estimate_spreads(
        self,
        prices: pd.Series,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
        trade_prices: Optional[pd.Series] = None,
        mid_prices: Optional[pd.Series] = None,
        date: Optional[datetime] = None,
    ) -> SpreadEstimate:
        """Compute all available spread estimates."""
        roll = self.roll_spread(prices)
        eff = 0.0
        if trade_prices is not None and mid_prices is not None:
            eff = self.effective_spread(trade_prices, mid_prices)
        quot = 0.0
        rel = 0.0
        if bid is not None and ask is not None:
            quot = self.quoted_spread(bid, ask)
            rel = self.relative_spread(bid, ask)
        return SpreadEstimate(
            date=date, roll_spread=roll, effective_spread=eff,
            quoted_spread=quot, relative_spread=rel,
        )

    def rolling_spread(
        self, prices: pd.Series, window: int = 21,
    ) -> pd.Series:
        """Rolling Roll spread estimate."""
        result = prices.rolling(window).apply(
            lambda x: self.roll_spread(x), raw=False,
        )
        return result

    # ------------------------------------------------------------------
    # 2. Price impact modelling
    # ------------------------------------------------------------------

    @staticmethod
    def kyle_lambda(
        price_changes: pd.Series,
        signed_volume: pd.Series,
    ) -> PriceImpact:
        """Kyle (1985) lambda: dp = lambda * signed_flow + eps.

        Args:
            price_changes: dp series.
            signed_volume: Positive for buys, negative for sells.
        """
        aligned = pd.DataFrame({"dp": price_changes, "sv": signed_volume}).dropna()
        if len(aligned) < 5:
            return PriceImpact()

        y = aligned["dp"].values
        x = aligned["sv"].values
        x_c = np.column_stack([np.ones(len(y)), x])
        try:
            betas, _, _, _ = np.linalg.lstsq(x_c, y, rcond=None)
        except np.linalg.LinAlgError:
            return PriceImpact()

        lam = float(betas[1])
        pred = x_c @ betas
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return PriceImpact(
            kyle_lambda=lam,
            permanent_impact=lam,
            temporary_impact=0.0,
            total_impact=lam,
            r_squared=r2,
        )

    @staticmethod
    def permanent_temporary_impact(
        price_changes: pd.Series,
        signed_volume: pd.Series,
        lag: int = 5,
    ) -> PriceImpact:
        """Decompose impact into permanent and temporary components.

        Permanent = correlation of signed_volume with price change over lag.
        Temporary = immediate impact minus permanent.
        """
        aligned = pd.DataFrame({"dp": price_changes, "sv": signed_volume}).dropna()
        if len(aligned) < lag + 5:
            return PriceImpact()

        y_imm = aligned["dp"].values
        x = aligned["sv"].values
        x_c = np.column_stack([np.ones(len(y_imm)), x])
        try:
            betas_imm, _, _, _ = np.linalg.lstsq(x_c, y_imm, rcond=None)
        except np.linalg.LinAlgError:
            return PriceImpact()
        total_lam = float(betas_imm[1])

        # Permanent: regress cumulative price change (lag periods ahead) on signed_volume
        cum_dp = aligned["dp"].rolling(lag).sum().shift(-lag + 1).dropna()
        sv_aligned = aligned["sv"].iloc[:len(cum_dp)]
        if len(cum_dp) < 5:
            return PriceImpact(kyle_lambda=total_lam, total_impact=total_lam)

        y_perm = cum_dp.values
        x_perm = np.column_stack([np.ones(len(y_perm)), sv_aligned.values])
        try:
            betas_perm, _, _, _ = np.linalg.lstsq(x_perm, y_perm, rcond=None)
        except np.linalg.LinAlgError:
            return PriceImpact(kyle_lambda=total_lam, total_impact=total_lam)
        perm_lam = float(betas_perm[1])
        temp_lam = total_lam - perm_lam

        pred = x_c @ betas_imm
        ss_res = float(((y_imm - pred) ** 2).sum())
        ss_tot = float(((y_imm - y_imm.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return PriceImpact(
            kyle_lambda=total_lam,
            permanent_impact=perm_lam,
            temporary_impact=temp_lam,
            total_impact=total_lam,
            r_squared=r2,
        )

    # ------------------------------------------------------------------
    # 3. Order flow toxicity
    # ------------------------------------------------------------------

    def compute_vpin(
        self,
        volume: pd.Series,
        price_changes: pd.Series,
    ) -> float:
        """Volume-synchronised Probability of Informed Trading (VPIN).

        Classifies volume bars as buy/sell using tick rule on price changes,
        then measures absolute imbalance across volume buckets.
        """
        aligned = pd.DataFrame({"vol": volume, "dp": price_changes}).dropna()
        if aligned.empty:
            return 0.0

        # Classify each bar as buy or sell
        buy_vol = aligned["vol"].where(aligned["dp"] > 0, 0.0)
        sell_vol = aligned["vol"].where(aligned["dp"] <= 0, 0.0)

        total = aligned["vol"].sum()
        if total <= 0:
            return 0.0

        n = self.n_vpin_buckets
        bucket_size = total / max(n, 1)
        if bucket_size <= 0:
            return 0.0

        # Accumulate into volume buckets
        cum_vol = 0.0
        cum_buy = 0.0
        cum_sell = 0.0
        imbalances: List[float] = []

        for i in range(len(aligned)):
            bv = float(buy_vol.iloc[i])
            sv = float(sell_vol.iloc[i])
            v = float(aligned["vol"].iloc[i])
            cum_buy += bv
            cum_sell += sv
            cum_vol += v

            while cum_vol >= bucket_size and len(imbalances) < n:
                imbalances.append(abs(cum_buy - cum_sell) / bucket_size)
                cum_vol -= bucket_size
                # Proportional reset
                if v > 0:
                    ratio = min(cum_vol / v, 1.0) if v > 0 else 0.0
                    cum_buy = bv * ratio
                    cum_sell = sv * ratio
                else:
                    cum_buy = 0.0
                    cum_sell = 0.0

        if not imbalances:
            return 0.0
        return float(np.mean(imbalances))

    @staticmethod
    def order_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> float:
        """Simple order imbalance: |buy - sell| / (buy + sell)."""
        total_buy = float(buy_volume.sum())
        total_sell = float(sell_volume.sum())
        total = total_buy + total_sell
        if total <= 0:
            return 0.0
        return abs(total_buy - total_sell) / total

    def compute_toxicity(
        self,
        volume: pd.Series,
        price_changes: pd.Series,
        buy_volume: Optional[pd.Series] = None,
        sell_volume: Optional[pd.Series] = None,
    ) -> ToxicityMetrics:
        """Full toxicity snapshot."""
        vpin = self.compute_vpin(volume, price_changes)
        oi = 0.0
        buy_r = 0.5
        if buy_volume is not None and sell_volume is not None:
            oi = self.order_imbalance(buy_volume, sell_volume)
            total = float(buy_volume.sum() + sell_volume.sum())
            buy_r = float(buy_volume.sum()) / total if total > 0 else 0.5

        if vpin >= self.toxicity_threshold:
            level = "toxic"
        elif vpin >= self.toxicity_threshold * 0.7:
            level = "elevated"
        elif vpin >= self.toxicity_threshold * 0.4:
            level = "normal"
        else:
            level = "low"

        return ToxicityMetrics(
            vpin=vpin, order_imbalance=oi,
            buy_ratio=buy_r, toxicity_level=level,
        )

    # ------------------------------------------------------------------
    # 4. Intraday volatility patterns
    # ------------------------------------------------------------------

    def intraday_patterns(
        self,
        returns: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
        spreads: Optional[pd.DataFrame] = None,
    ) -> List[IntradayPattern]:
        """Compute average volatility / volume / spread by intraday bucket.

        Args:
            returns: DataFrame with 'time_bucket' and 'return' columns,
                     OR a Series indexed by datetime with intraday granularity.
            volume: Optional matching volume data.
            spreads: Optional matching spread data.
        """
        if isinstance(returns, pd.Series):
            # Bucket by hour of day
            df = pd.DataFrame({
                "return": returns,
                "hour": returns.index.hour if hasattr(returns.index, "hour") else 0,
            })
            if volume is not None and isinstance(volume, pd.Series):
                df["volume"] = volume.reindex(returns.index).fillna(0)
            else:
                df["volume"] = 0.0
            if spreads is not None and isinstance(spreads, pd.Series):
                df["spread"] = spreads.reindex(returns.index).fillna(0)
            else:
                df["spread"] = 0.0

            results: List[IntradayPattern] = []
            for hour, grp in df.groupby("hour"):
                results.append(IntradayPattern(
                    bucket_label=f"{int(hour):02d}:00",
                    avg_volatility=float(grp["return"].std() * np.sqrt(TRADING_DAYS)),
                    avg_spread=float(grp["spread"].mean()),
                    avg_volume=float(grp["volume"].mean()),
                    n_observations=len(grp),
                ))
            return results

        # DataFrame with explicit time_bucket column
        if "time_bucket" not in returns.columns or "return" not in returns.columns:
            return []

        df = returns.copy()
        if volume is not None and "volume" in volume.columns:
            df["volume"] = volume["volume"].values[:len(df)]
        elif "volume" not in df.columns:
            df["volume"] = 0.0
        if spreads is not None and "spread" in spreads.columns:
            df["spread"] = spreads["spread"].values[:len(df)]
        elif "spread" not in df.columns:
            df["spread"] = 0.0

        results = []
        for bucket, grp in df.groupby("time_bucket"):
            results.append(IntradayPattern(
                bucket_label=str(bucket),
                avg_volatility=float(grp["return"].std() * np.sqrt(TRADING_DAYS)),
                avg_spread=float(grp["spread"].mean()),
                avg_volume=float(grp["volume"].mean()),
                n_observations=len(grp),
            ))
        return results

    @staticmethod
    def detect_u_shape(patterns: List[IntradayPattern]) -> bool:
        """Detect classic U-shaped intraday volatility pattern.

        True if first and last buckets have higher vol than middle.
        """
        if len(patterns) < 3:
            return False
        vols = [p.avg_volatility for p in patterns]
        n = len(vols)
        edge_avg = (vols[0] + vols[-1]) / 2.0
        mid_start = n // 4
        mid_end = 3 * n // 4
        mid_avg = np.mean(vols[mid_start:mid_end]) if mid_end > mid_start else 0.0
        return edge_avg > mid_avg * 1.05  # 5% higher at edges

    @staticmethod
    def overnight_gaps(
        open_prices: pd.Series,
        close_prices: pd.Series,
    ) -> List[OvernightGap]:
        """Compute overnight gap returns."""
        prev_close = close_prices.shift(1)
        gap = (open_prices / prev_close - 1).dropna()
        return [
            OvernightGap(date=dt, gap_return=float(g), abs_gap=abs(float(g)))
            for dt, g in gap.items()
        ]

    # ------------------------------------------------------------------
    # 5. Optimal execution windows
    # ------------------------------------------------------------------

    def optimal_execution_windows(
        self,
        patterns: List[IntradayPattern],
        top_n: int = 3,
    ) -> List[ExecutionWindow]:
        """Rank intraday buckets by execution quality.

        Quality = high volume, low spread. Score = volume_rank - spread_rank.
        """
        if not patterns:
            return []

        n = len(patterns)
        vols = [p.avg_volume for p in patterns]
        sprs = [p.avg_spread for p in patterns]

        # Rank: higher volume = better, lower spread = better
        vol_ranks = np.argsort(np.argsort(vols)).astype(float)  # ascending
        spr_ranks = np.argsort(np.argsort([-s for s in sprs])).astype(float)

        scores = vol_ranks + spr_ranks  # higher = better

        windows: List[ExecutionWindow] = []
        for i, p in enumerate(patterns):
            windows.append(ExecutionWindow(
                bucket_label=p.bucket_label,
                avg_spread=p.avg_spread,
                avg_volume=p.avg_volume,
                quality_score=float(scores[i]),
                rank=0,
            ))

        windows.sort(key=lambda w: w.quality_score, reverse=True)
        for i, w in enumerate(windows):
            w.rank = i + 1

        return windows[:top_n]

    # ------------------------------------------------------------------
    # 6. Maker vs taker analysis
    # ------------------------------------------------------------------

    @staticmethod
    def maker_taker_analysis(
        fills: pd.DataFrame,
    ) -> List[MakerTakerStats]:
        """Analyse fill quality by maker/taker side.

        Args:
            fills: DataFrame with columns: side ('maker'/'taker'),
                   price, midprice, spread (optional).
        """
        required = {"side", "price", "midprice"}
        if not required.issubset(fills.columns):
            return []

        results: List[MakerTakerStats] = []
        for side, grp in fills.groupby("side"):
            price_imp = (grp["midprice"] - grp["price"]).abs()
            avg_pi = float(price_imp.mean()) if len(grp) > 0 else 0.0

            avg_spread = 0.0
            if "spread" in grp.columns:
                avg_spread = float(grp["spread"].mean())

            fill_rate = len(grp) / len(fills) if len(fills) > 0 else 0.0

            results.append(MakerTakerStats(
                side=str(side),
                n_fills=len(grp),
                avg_price_improvement=avg_pi,
                avg_spread_captured=avg_spread,
                fill_rate=fill_rate,
            ))
        return results

    # ------------------------------------------------------------------
    # 7. HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 700, height: int = 200,
        color: str = "#2980b9",
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        y_min = min(values) * 0.9 if min(values) > 0 else min(values) * 1.1
        y_max = max(values) * 1.1
        if y_max <= y_min:
            y_max = y_min + 0.01
        pad_l, pad_r, pad_t, pad_b = 50, 15, 25, 25
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

        def tx(i: int) -> float:
            return pad_l + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph

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
    def _svg_heatmap(
        patterns: List[IntradayPattern],
        width: int = 700, height: int = 100,
    ) -> str:
        """Horizontal heatmap of execution quality by bucket."""
        if not patterns:
            return ""
        n = len(patterns)
        vols = [p.avg_volatility for p in patterns]
        v_min, v_max = min(vols), max(vols)
        if v_max == v_min:
            v_max = v_min + 0.01
        cell_w = (width - 20) / max(n, 1)
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="14" text-anchor="middle" font-size="11" '
                 f'font-weight="bold" fill="#1a1a2e">Execution Quality Heatmap (green=calm, red=volatile)</text>')
        for i, pat in enumerate(patterns):
            frac = (pat.avg_volatility - v_min) / (v_max - v_min)
            r = int(39 + frac * (231 - 39))
            g = int(174 - frac * (174 - 76))
            b = int(96 - frac * (96 - 60))
            x = 10 + i * cell_w
            p.append(f'<rect x="{x:.0f}" y="22" width="{cell_w:.0f}" height="45" '
                     f'fill="rgb({r},{g},{b})" rx="3"/>')
            p.append(f'<text x="{x + cell_w / 2:.0f}" y="80" text-anchor="middle" '
                     f'font-size="9" fill="#666">{pat.bucket_label}</text>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _toxicity_gauge(tox: ToxicityMetrics, width: int = 300, height: int = 80) -> str:
        """Gauge-style toxicity indicator."""
        colors = {"low": "#27ae60", "normal": "#2980b9", "elevated": "#e67e22", "toxic": "#e74c3c"}
        c = colors.get(tox.toxicity_level, "#999")
        bar_w = min(tox.vpin, 1.0) * (width - 40)
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">VPIN Toxicity</text>')
        p.append(f'<rect x="20" y="25" width="{width - 40}" height="20" fill="#eee" rx="4"/>')
        p.append(f'<rect x="20" y="25" width="{bar_w:.0f}" height="20" fill="{c}" rx="4"/>')
        p.append(f'<text x="{width // 2}" y="60" text-anchor="middle" font-size="11" '
                 f'fill="{c}" font-weight="bold">{tox.vpin:.2f} — {tox.toxicity_level.upper()}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        spread: Optional[SpreadEstimate] = None,
        impact: Optional[PriceImpact] = None,
        toxicity: Optional[ToxicityMetrics] = None,
        patterns: Optional[List[IntradayPattern]] = None,
        windows: Optional[List[ExecutionWindow]] = None,
        maker_taker: Optional[List[MakerTakerStats]] = None,
        rolling_spreads: Optional[List[float]] = None,
        output_path: str = "reports/microstructure.html",
    ) -> str:
        """HTML report: spread evolution, toxicity, execution heatmap."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Spread chart
        spread_svg = ""
        if rolling_spreads:
            spread_svg = self._svg_line(rolling_spreads, "Roll Spread Evolution", color="#2980b9")

        # Spread table
        spread_html = ""
        if spread is not None:
            spread_html = f"""
<h2>Bid-Ask Spread Estimates</h2>
<table class="metrics"><tr><th>Roll</th><th>Effective</th><th>Quoted</th><th>Relative</th></tr>
<tr><td>{spread.roll_spread:.6f}</td><td>{spread.effective_spread:.6f}</td>
<td>{spread.quoted_spread:.6f}</td><td>{spread.relative_spread:.4%}</td></tr></table>"""

        # Impact
        impact_html = ""
        if impact is not None:
            impact_html = f"""
<h2>Price Impact</h2>
<table class="metrics"><tr><th>Kyle &lambda;</th><th>Permanent</th><th>Temporary</th>
<th>Total</th><th>R&sup2;</th></tr>
<tr><td>{impact.kyle_lambda:.6f}</td><td>{impact.permanent_impact:.6f}</td>
<td>{impact.temporary_impact:.6f}</td><td>{impact.total_impact:.6f}</td>
<td>{impact.r_squared:.4f}</td></tr></table>"""

        # Toxicity gauge
        tox_svg = ""
        tox_html = ""
        if toxicity is not None:
            tox_svg = self._toxicity_gauge(toxicity)
            tox_html = f"""
<table class="metrics"><tr><th>VPIN</th><th>Order Imbalance</th><th>Buy Ratio</th><th>Level</th></tr>
<tr><td>{toxicity.vpin:.4f}</td><td>{toxicity.order_imbalance:.4f}</td>
<td>{toxicity.buy_ratio:.2%}</td><td>{toxicity.toxicity_level}</td></tr></table>"""

        # Heatmap
        heatmap_svg = ""
        if patterns:
            heatmap_svg = '<h2>Intraday Execution Quality</h2>\n' + self._svg_heatmap(patterns)

        # Execution windows
        win_html = ""
        if windows:
            rows = [
                f"<tr><td>{w.bucket_label}</td><td>{w.avg_spread:.6f}</td>"
                f"<td>{w.avg_volume:.0f}</td><td>{w.quality_score:.1f}</td>"
                f"<td>#{w.rank}</td></tr>"
                for w in windows
            ]
            win_html = f"""
<h2>Optimal Execution Windows</h2>
<table><tr><th>Bucket</th><th>Avg Spread</th><th>Avg Volume</th>
<th>Quality Score</th><th>Rank</th></tr>
{''.join(rows)}</table>"""

        # Maker/taker
        mt_html = ""
        if maker_taker:
            rows = [
                f"<tr><td>{m.side}</td><td>{m.n_fills}</td>"
                f"<td>{m.avg_price_improvement:.6f}</td>"
                f"<td>{m.avg_spread_captured:.6f}</td>"
                f"<td>{m.fill_rate:.1%}</td></tr>"
                for m in maker_taker
            ]
            mt_html = f"""
<h2>Maker vs Taker Analysis</h2>
<table><tr><th>Side</th><th>Fills</th><th>Avg Price Improvement</th>
<th>Avg Spread Captured</th><th>Fill Rate</th></tr>
{''.join(rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Microstructure Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.metrics {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.charts {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
</style></head><body>
<h1>Market Microstructure Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

{spread_svg}
{spread_html}
{impact_html}

<h2>Order Flow Toxicity</h2>
<div class="charts">{tox_svg}</div>
{tox_html}

{heatmap_svg}
{win_html}
{mt_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Microstructure report -> %s", path)
        return str(path)
