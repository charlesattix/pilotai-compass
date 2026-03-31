"""
Institutional trade flow analyzer.

Detects large block trades, classifies smart-money vs retail flow,
computes flow toxicity (VPIN model), identifies institutional
accumulation/distribution, generates order-flow imbalance signals
and flow momentum/reversal signals.

Generates an HTML report at reports/trade_flow.html with flow heatmap,
toxicity gauge, smart-money indicator, and cumulative flow chart.

Usage::

    from compass.trade_flow import TradeFlowAnalyzer
    analyzer = TradeFlowAnalyzer(trades_df)
    results = analyzer.analyze()
    analyzer.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "trade_flow.html"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class BlockTrade:
    """A detected large block trade."""
    timestamp: str
    price: float
    volume: int
    side: str               # "buy" or "sell"
    size_multiple: float    # vs median trade size
    price_impact: float     # signed % move after


@dataclass
class FlowClassification:
    """Smart-money vs retail flow classification for one bucket."""
    bucket: str             # time bucket label
    smart_volume: float
    retail_volume: float
    smart_ratio: float      # smart / total
    net_smart_flow: float   # smart buys - smart sells
    net_retail_flow: float


@dataclass
class VPINResult:
    """Volume-synchronized probability of informed trading."""
    vpin: float             # 0-1
    percentile: float       # historical percentile
    toxicity_level: str     # "low", "medium", "high", "extreme"
    bucket_size: int
    n_buckets: int


@dataclass
class AccumulationSignal:
    """Institutional accumulation/distribution detection."""
    phase: str              # "accumulation", "distribution", "neutral"
    strength: float         # 0-1
    duration_bars: int
    smart_net_flow: float
    price_trend: str        # "up", "down", "flat"
    divergence: bool        # price/flow divergence


@dataclass
class FlowImbalance:
    """Order flow imbalance signal."""
    imbalance: float        # -1 to +1 (negative = sell pressure)
    abs_imbalance: float
    buy_volume: float
    sell_volume: float
    signal: str             # "strong_buy", "buy", "neutral", "sell", "strong_sell"


@dataclass
class FlowMomentum:
    """Flow momentum and reversal signals."""
    momentum: float         # positive = buy momentum
    acceleration: float     # change in momentum
    signal: str             # "momentum_buy", "momentum_sell", "reversal_buy", "reversal_sell", "neutral"
    lookback: int


@dataclass
class FlowSnapshot:
    """Complete flow state at one time bucket."""
    bucket: str
    buy_volume: float
    sell_volume: float
    net_flow: float
    cumulative_flow: float
    vpin: float
    imbalance: float
    smart_ratio: float


# ── VPIN computation ────────────────────────────────────────────────────


def compute_vpin(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    bucket_size: int = 50,
) -> Tuple[float, np.ndarray]:
    """Compute Volume-synchronized Probability of Informed Trading.

    VPIN = mean(|V_buy - V_sell|) / (V_buy + V_sell) over volume buckets.
    Returns (current_vpin, vpin_series).
    """
    total = buy_volume + sell_volume
    if len(total) < bucket_size or total.sum() == 0:
        return 0.0, np.array([0.0])

    # Volume buckets: aggregate trades until bucket_size volume reached
    abs_imb = np.abs(buy_volume - sell_volume)
    total_vol = buy_volume + sell_volume

    n = len(total)
    vpins = []
    for end in range(bucket_size, n + 1, max(1, bucket_size // 5)):
        start = max(0, end - bucket_size)
        window_imb = abs_imb[start:end].sum()
        window_vol = total_vol[start:end].sum()
        if window_vol > 0:
            vpins.append(window_imb / window_vol)

    if not vpins:
        return 0.0, np.array([0.0])

    vpin_arr = np.array(vpins)
    return float(vpin_arr[-1]), vpin_arr


# ── Trade classification (Lee-Ready approximation) ──────────────────────


def classify_trades(
    prices: np.ndarray,
    volumes: np.ndarray,
    mid_prices: Optional[np.ndarray] = None,
    block_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify trades as buy/sell using tick rule + size.

    Returns (buy_volume, sell_volume, is_block) arrays.
    """
    n = len(prices)
    buy_vol = np.zeros(n)
    sell_vol = np.zeros(n)
    is_block = np.zeros(n, dtype=bool)

    if mid_prices is not None:
        # Quote rule: above mid = buy, below = sell
        buy_mask = prices > mid_prices
        sell_mask = prices < mid_prices
        eq_mask = prices == mid_prices
    else:
        # Tick rule: uptick = buy, downtick = sell
        diffs = np.diff(prices, prepend=prices[0])
        buy_mask = diffs > 0
        sell_mask = diffs < 0
        eq_mask = diffs == 0

    buy_vol[buy_mask] = volumes[buy_mask]
    sell_vol[sell_mask] = volumes[sell_mask]
    # Equal: split 50/50
    buy_vol[eq_mask] = volumes[eq_mask] * 0.5
    sell_vol[eq_mask] = volumes[eq_mask] * 0.5

    # Block detection
    median_vol = np.median(volumes[volumes > 0]) if (volumes > 0).any() else 1.0
    is_block = volumes > median_vol * block_threshold

    return buy_vol, sell_vol, is_block


# ── Smart money classification ──────────────────────────────────────────


def classify_smart_money(
    volumes: np.ndarray,
    is_block: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    """Classify trades as smart-money (True) or retail (False).

    Heuristic: block trades + trades at extremes of price range.
    """
    n = len(volumes)
    is_smart = np.zeros(n, dtype=bool)

    # Block trades → smart money
    is_smart |= is_block

    # Trades in top/bottom 10% of price range → likely institutional
    if n >= 20:
        rolling_hi = pd.Series(prices).rolling(20, min_periods=1).max().values
        rolling_lo = pd.Series(prices).rolling(20, min_periods=1).min().values
        rng = rolling_hi - rolling_lo
        rng[rng < 1e-10] = 1.0
        position = (prices - rolling_lo) / rng
        is_smart |= (position > 0.9) | (position < 0.1)

    return is_smart


# ── Analyzer ────────────────────────────────────────────────────────────


class TradeFlowAnalyzer:
    """Institutional trade flow analysis."""

    def __init__(
        self,
        trades: pd.DataFrame,
        price_col: str = "price",
        volume_col: str = "volume",
        mid_col: Optional[str] = None,
        bucket_freq: str = "5min",
        vpin_buckets: int = 50,
        block_threshold: float = 5.0,
        lookback: int = 20,
    ) -> None:
        self.trades = trades.copy()
        self.price_col = price_col
        self.volume_col = volume_col
        self.mid_col = mid_col
        self.bucket_freq = bucket_freq
        self.vpin_buckets = vpin_buckets
        self.block_threshold = block_threshold
        self.lookback = lookback

        # Ensure columns
        if price_col not in self.trades.columns:
            raise ValueError(f"Price column {price_col!r} not in DataFrame")
        if volume_col not in self.trades.columns:
            self.trades[volume_col] = 100

        # Results
        self.block_trades: List[BlockTrade] = []
        self.flow_classification: List[FlowClassification] = []
        self.vpin_result: Optional[VPINResult] = None
        self.accumulation: Optional[AccumulationSignal] = None
        self.imbalance: Optional[FlowImbalance] = None
        self.momentum: Optional[FlowMomentum] = None
        self.snapshots: List[FlowSnapshot] = []

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "TradeFlowAnalyzer":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a compact summary dict suitable for dashboards/alerting."""
        if self.vpin_result is None:
            self.analyze()
        vp = self.vpin_result
        acc = self.accumulation
        imb = self.imbalance
        mom = self.momentum
        return {
            "vpin": vp.vpin if vp else 0.0,
            "toxicity": vp.toxicity_level if vp else "unknown",
            "accumulation_phase": acc.phase if acc else "neutral",
            "accumulation_strength": acc.strength if acc else 0.0,
            "divergence": acc.divergence if acc else False,
            "imbalance": imb.imbalance if imb else 0.0,
            "imbalance_signal": imb.signal if imb else "neutral",
            "momentum_signal": mom.signal if mom else "neutral",
            "n_blocks": len(self.block_trades),
            "smart_ratio": float(np.mean([f.smart_ratio for f in self.flow_classification])) if self.flow_classification else 0.0,
        }

    def analyze(self) -> Dict[str, Any]:
        prices = self.trades[self.price_col].values.astype(float)
        volumes = self.trades[self.volume_col].values.astype(float)
        mids = self.trades[self.mid_col].values.astype(float) if self.mid_col and self.mid_col in self.trades.columns else None

        buy_vol, sell_vol, is_block = classify_trades(prices, volumes, mids, self.block_threshold)
        is_smart = classify_smart_money(volumes, is_block, prices)

        self.block_trades = self._detect_blocks(prices, volumes, is_block, buy_vol, sell_vol)
        self.flow_classification = self._classify_flow_buckets(buy_vol, sell_vol, is_smart)
        self.vpin_result = self._compute_vpin(buy_vol, sell_vol)
        self.snapshots = self._build_snapshots(buy_vol, sell_vol, is_smart)
        self.accumulation = self._detect_accumulation(buy_vol, sell_vol, is_smart, prices)
        self.imbalance = self._flow_imbalance(buy_vol, sell_vol)
        self.momentum = self._flow_momentum(buy_vol, sell_vol)

        return {
            "block_trades": self.block_trades,
            "flow_classification": self.flow_classification,
            "vpin": self.vpin_result,
            "accumulation": self.accumulation,
            "imbalance": self.imbalance,
            "momentum": self.momentum,
            "snapshots": self.snapshots,
        }

    # ── Block detection ─────────────────────────────────────────────────

    def _detect_blocks(
        self, prices: np.ndarray, volumes: np.ndarray,
        is_block: np.ndarray, buy_vol: np.ndarray, sell_vol: np.ndarray,
    ) -> List[BlockTrade]:
        median_vol = float(np.median(volumes[volumes > 0])) if (volumes > 0).any() else 1.0
        blocks: List[BlockTrade] = []
        idx = self.trades.index

        for i in np.where(is_block)[0]:
            side = "buy" if buy_vol[i] > sell_vol[i] else "sell"
            mult = float(volumes[i] / median_vol)

            # Price impact: % move over next 5 trades
            future = prices[i + 1: i + 6]
            impact = float((future.mean() / prices[i] - 1) * 100) if len(future) > 0 else 0.0

            blocks.append(BlockTrade(
                timestamp=str(idx[i]),
                price=float(prices[i]),
                volume=int(volumes[i]),
                side=side,
                size_multiple=mult,
                price_impact=impact,
            ))
        return sorted(blocks, key=lambda b: -b.size_multiple)[:100]

    # ── Flow classification by bucket ───────────────────────────────────

    def _classify_flow_buckets(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
        is_smart: np.ndarray,
    ) -> List[FlowClassification]:
        n = len(buy_vol)
        bucket_size = max(1, n // 20)
        results: List[FlowClassification] = []

        for start in range(0, n, bucket_size):
            end = min(start + bucket_size, n)
            bv = buy_vol[start:end]
            sv = sell_vol[start:end]
            sm = is_smart[start:end]

            smart_vol = float((bv[sm].sum() + sv[sm].sum()))
            retail_vol = float((bv[~sm].sum() + sv[~sm].sum()))
            total = smart_vol + retail_vol
            ratio = smart_vol / total if total > 0 else 0.0

            net_smart = float(bv[sm].sum() - sv[sm].sum())
            net_retail = float(bv[~sm].sum() - sv[~sm].sum())

            results.append(FlowClassification(
                bucket=f"B{start // bucket_size + 1}",
                smart_volume=smart_vol, retail_volume=retail_vol,
                smart_ratio=ratio,
                net_smart_flow=net_smart, net_retail_flow=net_retail,
            ))
        return results

    # ── VPIN ────────────────────────────────────────────────────────────

    def _compute_vpin(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
    ) -> VPINResult:
        vpin, vpin_series = compute_vpin(buy_vol, sell_vol, self.vpin_buckets)

        if len(vpin_series) > 1:
            pct = float((vpin_series < vpin).mean())
        else:
            pct = 0.5

        if vpin > 0.7:
            level = "extreme"
        elif vpin > 0.5:
            level = "high"
        elif vpin > 0.3:
            level = "medium"
        else:
            level = "low"

        return VPINResult(
            vpin=vpin, percentile=pct, toxicity_level=level,
            bucket_size=self.vpin_buckets, n_buckets=len(vpin_series),
        )

    # ── Snapshots ───────────────────────────────────────────────────────

    def _build_snapshots(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
        is_smart: np.ndarray,
    ) -> List[FlowSnapshot]:
        n = len(buy_vol)
        bucket_size = max(1, n // 50)
        snapshots: List[FlowSnapshot] = []
        cumulative = 0.0

        for start in range(0, n, bucket_size):
            end = min(start + bucket_size, n)
            bv = float(buy_vol[start:end].sum())
            sv = float(sell_vol[start:end].sum())
            net = bv - sv
            cumulative += net
            total = bv + sv

            # Per-bucket VPIN
            abs_imb = float(np.abs(buy_vol[start:end] - sell_vol[start:end]).sum())
            vpin = abs_imb / total if total > 0 else 0.0

            imb = net / total if total > 0 else 0.0
            sm = is_smart[start:end]
            smart_vol = float((buy_vol[start:end][sm].sum() + sell_vol[start:end][sm].sum()))
            ratio = smart_vol / total if total > 0 else 0.0

            snapshots.append(FlowSnapshot(
                bucket=f"B{start // bucket_size + 1}",
                buy_volume=bv, sell_volume=sv, net_flow=net,
                cumulative_flow=cumulative, vpin=vpin,
                imbalance=imb, smart_ratio=ratio,
            ))
        return snapshots

    # ── Accumulation/distribution ───────────────────────────────────────

    def _detect_accumulation(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
        is_smart: np.ndarray, prices: np.ndarray,
    ) -> AccumulationSignal:
        n = len(buy_vol)
        lb = min(self.lookback, n)
        recent_smart_buy = float(buy_vol[-lb:][is_smart[-lb:]].sum())
        recent_smart_sell = float(sell_vol[-lb:][is_smart[-lb:]].sum())
        net_smart = recent_smart_buy - recent_smart_sell
        total_smart = recent_smart_buy + recent_smart_sell

        strength = abs(net_smart) / (total_smart + 1e-10)
        strength = min(strength, 1.0)

        # Price trend
        if lb >= 5:
            price_change = float(prices[-1] / prices[-lb] - 1)
        else:
            price_change = 0.0
        if price_change > 0.005:
            price_trend = "up"
        elif price_change < -0.005:
            price_trend = "down"
        else:
            price_trend = "flat"

        if net_smart > 0 and strength > 0.1:
            phase = "accumulation"
        elif net_smart < 0 and strength > 0.1:
            phase = "distribution"
        else:
            phase = "neutral"

        # Divergence: smart money buying while price falling, or vice versa
        divergence = (phase == "accumulation" and price_trend == "down") or \
                     (phase == "distribution" and price_trend == "up")

        return AccumulationSignal(
            phase=phase, strength=strength,
            duration_bars=lb, smart_net_flow=net_smart,
            price_trend=price_trend, divergence=divergence,
        )

    # ── Flow imbalance ──────────────────────────────────────────────────

    def _flow_imbalance(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
    ) -> FlowImbalance:
        lb = min(self.lookback, len(buy_vol))
        bv = float(buy_vol[-lb:].sum())
        sv = float(sell_vol[-lb:].sum())
        total = bv + sv
        imb = (bv - sv) / total if total > 0 else 0.0

        if imb > 0.3:
            signal = "strong_buy"
        elif imb > 0.1:
            signal = "buy"
        elif imb < -0.3:
            signal = "strong_sell"
        elif imb < -0.1:
            signal = "sell"
        else:
            signal = "neutral"

        return FlowImbalance(
            imbalance=imb, abs_imbalance=abs(imb),
            buy_volume=bv, sell_volume=sv, signal=signal,
        )

    # ── Flow momentum ───────────────────────────────────────────────────

    def _flow_momentum(
        self, buy_vol: np.ndarray, sell_vol: np.ndarray,
    ) -> FlowMomentum:
        n = len(buy_vol)
        lb = min(self.lookback, n)
        half = lb // 2

        if lb < 4:
            return FlowMomentum(0, 0, "neutral", lb)

        net = buy_vol - sell_vol
        recent = float(net[-half:].sum())
        earlier = float(net[-lb:-half].sum())

        momentum = recent - earlier
        # Acceleration: compare two halves of the recent period
        q1 = float(net[-half: -half // 2].sum()) if half > 1 else 0
        q2 = float(net[-half // 2:].sum()) if half > 1 else 0
        acceleration = q2 - q1

        # Signal
        if momentum > 0 and acceleration > 0:
            signal = "momentum_buy"
        elif momentum < 0 and acceleration < 0:
            signal = "momentum_sell"
        elif momentum > 0 and acceleration < 0:
            signal = "reversal_sell"
        elif momentum < 0 and acceleration > 0:
            signal = "reversal_buy"
        else:
            signal = "neutral"

        return FlowMomentum(
            momentum=float(momentum), acceleration=float(acceleration),
            signal=signal, lookback=lb,
        )

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.vpin_result is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        return {
            "cumulative_flow": self._chart_cumulative(),
            "flow_heatmap": self._chart_heatmap(),
            "toxicity_gauge": self._chart_toxicity(),
            "smart_money": self._chart_smart_money(),
        }

    def _chart_cumulative(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.snapshots:
            return ""
        xs = range(len(self.snapshots))
        cum = [s.cumulative_flow for s in self.snapshots]
        net = [s.net_flow for s in self.snapshots]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(xs, cum, color="#3b82f6", lw=1.2)
        ax1.fill_between(xs, cum, alpha=0.1, color="#3b82f6")
        ax1.set_ylabel("Cumulative Flow"); ax1.set_title("Cumulative Net Flow", fontsize=11)
        ax1.axhline(0, color="black", lw=0.5); ax1.grid(True, alpha=0.2)
        colors = ["#16a34a" if n > 0 else "#dc2626" for n in net]
        ax2.bar(xs, net, color=colors, alpha=0.7, width=1.0)
        ax2.set_ylabel("Net Flow"); ax2.set_xlabel("Bucket"); ax2.set_title("Net Flow per Bucket", fontsize=10)
        ax2.axhline(0, color="black", lw=0.5); ax2.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_heatmap(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.snapshots:
            return ""
        n = len(self.snapshots)
        metrics = np.zeros((4, n))
        for i, s in enumerate(self.snapshots):
            metrics[0, i] = s.imbalance
            metrics[1, i] = s.vpin
            metrics[2, i] = s.smart_ratio
            metrics[3, i] = s.net_flow / (s.buy_volume + s.sell_volume + 1e-10)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.15), 3))
        im = ax.imshow(metrics, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["Imbalance", "VPIN", "Smart Ratio", "Net/Total"], fontsize=8)
        ax.set_xlabel("Bucket"); ax.set_title("Flow Heatmap", fontsize=11)
        fig.colorbar(im, shrink=0.8); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_toxicity(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self.vpin_result is None:
            return ""
        fig, ax = plt.subplots(figsize=(4, 3))
        v = self.vpin_result.vpin
        color = "#16a34a" if v < 0.3 else "#f59e0b" if v < 0.5 else "#dc2626"
        ax.barh(["VPIN"], [v], color=color, alpha=0.85, height=0.4)
        ax.set_xlim(0, 1); ax.set_xlabel("Toxicity")
        ax.text(min(v + 0.03, 0.95), 0, f"{v:.2f}", va="center", fontsize=12, fontweight="bold")
        ax.set_title(f"Flow Toxicity: {self.vpin_result.toxicity_level.upper()}", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_smart_money(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.flow_classification:
            return ""
        buckets = [f.bucket for f in self.flow_classification]
        smart = [f.net_smart_flow for f in self.flow_classification]
        retail = [f.net_retail_flow for f in self.flow_classification]
        x = np.arange(len(buckets))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(7, len(buckets) * 0.4), 4))
        ax.bar(x - w/2, smart, w, label="Smart Money", color="#3b82f6", alpha=0.85)
        ax.bar(x + w/2, retail, w, label="Retail", color="#f59e0b", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(buckets, fontsize=7, rotation=45)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Net Flow"); ax.set_title("Smart Money vs Retail Flow", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        vp = self.vpin_result or VPINResult(0, 0, "low", 0, 0)
        acc = self.accumulation or AccumulationSignal("neutral", 0, 0, 0, "flat", False)
        imb = self.imbalance or FlowImbalance(0, 0, 0, 0, "neutral")
        mom = self.momentum or FlowMomentum(0, 0, "neutral", 0)

        tox_color = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626", "extreme": "#7f1d1d"}
        tc = tox_color.get(vp.toxicity_level, "#64748b")
        acc_cls = "good" if acc.phase == "accumulation" else "bad" if acc.phase == "distribution" else ""
        imb_cls = "good" if "buy" in imb.signal else "bad" if "sell" in imb.signal else ""

        block_rows = ""
        for b in self.block_trades[:20]:
            block_rows += (f'<tr><td>{b.timestamp}</td><td>{b.side}</td><td>{b.volume:,}</td>'
                           f'<td>{b.size_multiple:.1f}x</td><td>{b.price:.2f}</td>'
                           f'<td>{b.price_impact:+.3f}%</td></tr>\n')
        if not block_rows:
            block_rows = '<tr><td colspan="6" style="text-align:center;color:#64748b">No block trades</td></tr>'

        flow_rows = ""
        for f in self.flow_classification:
            flow_rows += (f'<tr><td>{f.bucket}</td><td>{f.smart_volume:,.0f}</td><td>{f.retail_volume:,.0f}</td>'
                          f'<td>{f.smart_ratio:.0%}</td><td>{f.net_smart_flow:+,.0f}</td><td>{f.net_retail_flow:+,.0f}</td></tr>\n')

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Institutional Trade Flow Analysis</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  .risk-badge {{ display:inline-block; padding:0.3em 0.8em; border-radius:4px; color:white; font-weight:700; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Institutional Trade Flow Analysis</h1>
<div class="meta">{len(self.trades)} trades &middot; {len(self.block_trades)} blocks &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value"><span class="risk-badge" style="background:{tc}">{vp.toxicity_level.upper()}</span></div><div class="label">VPIN Toxicity</div></div>
  <div class="kpi"><div class="value">{vp.vpin:.2f}</div><div class="label">VPIN</div></div>
  <div class="kpi"><div class="value {acc_cls}">{acc.phase}</div><div class="label">Accumulation</div></div>
  <div class="kpi"><div class="value {imb_cls}">{imb.signal}</div><div class="label">Flow Imbalance</div></div>
  <div class="kpi"><div class="value">{mom.signal}</div><div class="label">Momentum</div></div>
  <div class="kpi"><div class="value">{len(self.block_trades)}</div><div class="label">Block Trades</div></div>
</div>
<h2>1. Cumulative Flow</h2>{_img("cumulative_flow")}
<h2>2. Flow Heatmap</h2>{_img("flow_heatmap")}
<h2>3. Flow Toxicity (VPIN)</h2>{_img("toxicity_gauge")}
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>VPIN</td><td>{vp.vpin:.4f}</td></tr>
<tr><td>Percentile</td><td>{vp.percentile:.0%}</td></tr>
<tr><td>Buckets</td><td>{vp.n_buckets}</td></tr>
</tbody></table>
<h2>4. Smart Money vs Retail</h2>{_img("smart_money")}
<table><thead><tr><th>Bucket</th><th>Smart Vol</th><th>Retail Vol</th><th>Smart %</th><th>Net Smart</th><th>Net Retail</th></tr></thead>
<tbody>{flow_rows}</tbody></table>
<h2>5. Block Trades</h2>
<table><thead><tr><th>Time</th><th>Side</th><th>Volume</th><th>Size</th><th>Price</th><th>Impact</th></tr></thead>
<tbody>{block_rows}</tbody></table>
<h2>6. Accumulation / Distribution</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Phase</td><td class="{acc_cls}">{acc.phase}</td></tr>
<tr><td>Strength</td><td>{acc.strength:.2f}</td></tr>
<tr><td>Smart Net Flow</td><td>{acc.smart_net_flow:+,.0f}</td></tr>
<tr><td>Price Trend</td><td>{acc.price_trend}</td></tr>
<tr><td>Divergence</td><td class="{"bad" if acc.divergence else ""}">{acc.divergence}</td></tr>
</tbody></table>
<h2>7. Flow Signals</h2>
<table><thead><tr><th>Signal</th><th>Value</th></tr></thead><tbody>
<tr><td>Imbalance</td><td class="{imb_cls}">{imb.imbalance:+.3f} ({imb.signal})</td></tr>
<tr><td>Momentum</td><td>{mom.momentum:+,.0f} ({mom.signal})</td></tr>
<tr><td>Acceleration</td><td>{mom.acceleration:+,.0f}</td></tr>
</tbody></table>
<footer>Generated by <code>compass/trade_flow.py</code></footer>
</body></html>"""
        return html
