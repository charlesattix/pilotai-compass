"""
Automated signal research and discovery engine.

Generates signals from price data, screens them, filters correlated
ones, and ranks survivors by information coefficient and stability.

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


@dataclass
class SignalDefinition:
    name: str
    category: str         # momentum / mean_reversion / breakout / carry / value
    params: Dict = field(default_factory=dict)


@dataclass
class SignalMetrics:
    name: str
    ic: float             # information coefficient (rank corr with fwd returns)
    ic_std: float
    turnover: float       # daily turnover rate
    decay_halflife: float # autocorrelation half-life (days)
    sharpe: float
    passed_screen: bool


@dataclass
class SignalResearchResult:
    generated: int
    screened: int
    passed: int
    after_corr_filter: int
    top_signals: List[SignalMetrics]


class SignalResearcher:
    """Signal research engine.

    Args:
        ic_threshold: Minimum IC to pass screen.
        max_turnover: Maximum daily turnover.
        min_halflife: Minimum signal decay half-life (days).
        max_correlation: Max pairwise corr before dropping.
    """

    def __init__(
        self,
        ic_threshold: float = 0.02,
        max_turnover: float = 0.50,
        min_halflife: float = 5.0,
        max_correlation: float = 0.70,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.max_turnover = max_turnover
        self.min_halflife = min_halflife
        self.max_correlation = max_correlation

    # ------------------------------------------------------------------
    # Signal generators
    # ------------------------------------------------------------------

    @staticmethod
    def generate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-8)
        rsi = 100 - 100 / (1 + rs)
        return (rsi - 50) / 50  # normalised to [-1, 1]

    @staticmethod
    def generate_bollinger(prices: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + n_std * std
        lower = ma - n_std * std
        return (prices - ma) / (upper - lower + 1e-8)

    @staticmethod
    def generate_keltner(prices: pd.Series, period: int = 20, atr_mult: float = 1.5) -> pd.Series:
        ma = prices.rolling(period).mean()
        atr = prices.diff().abs().rolling(period).mean()
        upper = ma + atr_mult * atr
        lower = ma - atr_mult * atr
        return (prices - ma) / (upper - lower + 1e-8)

    @staticmethod
    def generate_donchian(prices: pd.Series, period: int = 20) -> pd.Series:
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        mid = (high + low) / 2
        return (prices - mid) / (high - low + 1e-8)

    @staticmethod
    def generate_momentum(prices: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
        fast_ret = prices.pct_change(fast)
        slow_ret = prices.pct_change(slow)
        return (fast_ret - slow_ret).clip(-1, 1)

    def generate_all(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Generate full signal zoo from price data."""
        signals = {}
        for p in [7, 14, 21]:
            signals[f"rsi_{p}"] = self.generate_rsi(prices, p)
        for p in [10, 20, 30]:
            signals[f"boll_{p}"] = self.generate_bollinger(prices, p)
        for p in [10, 20]:
            signals[f"keltner_{p}"] = self.generate_keltner(prices, p)
        for p in [10, 20, 50]:
            signals[f"donchian_{p}"] = self.generate_donchian(prices, p)
        for f, s in [(5, 20), (10, 50), (20, 100)]:
            signals[f"mom_{f}_{s}"] = self.generate_momentum(prices, f, s)
        return signals

    # ------------------------------------------------------------------
    # Signal metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ic(signal: pd.Series, forward_returns: pd.Series) -> Tuple[float, float]:
        aligned = pd.DataFrame({"sig": signal, "fwd": forward_returns}).dropna()
        if len(aligned) < 20:
            return 0.0, 1.0
        ic = float(aligned["sig"].corr(aligned["fwd"], method="spearman"))
        # Rolling IC std
        roll_ic = aligned["sig"].rolling(63).corr(aligned["fwd"]).dropna()
        ic_std = float(roll_ic.std()) if len(roll_ic) > 5 else 1.0
        return ic, ic_std

    @staticmethod
    def compute_turnover(signal: pd.Series) -> float:
        changes = signal.diff().abs().dropna()
        return float(changes.mean()) if not changes.empty else 0.0

    @staticmethod
    def compute_halflife(signal: pd.Series) -> float:
        ac = signal.dropna()
        if len(ac) < 10:
            return 0.0
        autocorr = float(ac.autocorr(lag=1))
        if autocorr <= 0:
            return 0.0
        return -1.0 / np.log(abs(autocorr)) if abs(autocorr) < 1 else float("inf")

    def evaluate_signal(
        self, name: str, signal: pd.Series, forward_returns: pd.Series,
    ) -> SignalMetrics:
        ic, ic_std = self.compute_ic(signal, forward_returns)
        turnover = self.compute_turnover(signal)
        halflife = self.compute_halflife(signal)
        # Quick Sharpe: sign(signal) * fwd_ret
        aligned = pd.DataFrame({"sig": signal, "fwd": forward_returns}).dropna()
        strat_ret = np.sign(aligned["sig"]) * aligned["fwd"]
        mu = float(strat_ret.mean())
        std = float(strat_ret.std())
        sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0

        passed = (abs(ic) >= self.ic_threshold
                  and turnover <= self.max_turnover
                  and halflife >= self.min_halflife)

        return SignalMetrics(name=name, ic=ic, ic_std=ic_std, turnover=turnover,
                              decay_halflife=halflife, sharpe=sharpe, passed_screen=passed)

    # ------------------------------------------------------------------
    # Correlation filtering
    # ------------------------------------------------------------------

    def correlation_filter(
        self, signals: Dict[str, pd.Series], passed_names: List[str],
    ) -> List[str]:
        """Drop highly correlated signals, keeping highest IC first."""
        if len(passed_names) <= 1:
            return list(passed_names)

        df = pd.DataFrame({n: signals[n] for n in passed_names if n in signals}).dropna()
        if df.empty:
            return list(passed_names)

        corr = df.corr().abs()
        kept: List[str] = []
        for name in passed_names:
            if name not in corr.columns:
                continue
            drop = False
            for existing in kept:
                if existing in corr.columns and corr.loc[name, existing] > self.max_correlation:
                    drop = True
                    break
            if not drop:
                kept.append(name)
        return kept

    # ------------------------------------------------------------------
    # Full research pipeline
    # ------------------------------------------------------------------

    def research(
        self, prices: pd.Series, forward_horizon: int = 5,
    ) -> SignalResearchResult:
        signals = self.generate_all(prices)
        fwd = prices.pct_change(forward_horizon).shift(-forward_horizon)

        all_metrics: List[SignalMetrics] = []
        for name, sig in signals.items():
            m = self.evaluate_signal(name, sig, fwd)
            all_metrics.append(m)

        passed = [m for m in all_metrics if m.passed_screen]
        passed.sort(key=lambda m: abs(m.ic), reverse=True)
        passed_names = [m.name for m in passed]

        after_corr = self.correlation_filter(signals, passed_names)
        top = [m for m in passed if m.name in after_corr]

        return SignalResearchResult(
            generated=len(signals), screened=len(all_metrics),
            passed=len(passed), after_corr_filter=len(after_corr),
            top_signals=top,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, result: SignalResearchResult,
        output_path: str = "reports/signal_research.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            f"<tr><td>{s.name}</td><td>{s.ic:.4f}</td><td>{s.turnover:.4f}</td>"
            f"<td>{s.decay_halflife:.1f}</td><td>{s.sharpe:.2f}</td></tr>"
            for s in result.top_signals
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Signal Research</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Signal Research Report</h1>
<div class="summary">
<p>Generated: {result.generated} | Screened: {result.screened} |
   Passed: {result.passed} | After Corr Filter: {result.after_corr_filter}</p>
</div>
<h2>Top Signals</h2>
<table><tr><th>Name</th><th>IC</th><th>Turnover</th><th>Half-Life</th><th>Sharpe</th></tr>
{''.join(rows)}</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
