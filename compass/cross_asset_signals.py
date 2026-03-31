"""
Cross-asset signal generator for multi-asset alpha.

Inter-market signals (equity-bond, equity-commodity, currency-equity),
lead-lag detection between asset pairs, cointegration testing
(Engle-Granger with ADF), spread trading signals (z-score), cross-asset
momentum (relative strength), and macro regime signals from yield curve /
credit spreads / VIX term structure.

Generates an HTML report at reports/cross_asset_signals.html with
correlation heatmap, lead-lag matrix, spread z-scores, and signal
dashboard.

Usage::

    from compass.cross_asset_signals import CrossAssetSignalGenerator
    gen = CrossAssetSignalGenerator(prices_df)
    results = gen.analyze()
    gen.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "cross_asset_signals.html"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class InterMarketCorrelation:
    """Correlation between two assets."""
    asset_a: str
    asset_b: str
    correlation: float
    rolling_corr_mean: float
    rolling_corr_std: float
    current_corr: float
    z_score: float             # how unusual current corr is


@dataclass
class LeadLagResult:
    """Lead-lag relationship between two assets."""
    leader: str
    lagger: str
    optimal_lag: int           # days
    correlation_at_lag: float
    p_value: float
    direction: str             # "positive" or "negative"


@dataclass
class CointegrationResult:
    """Engle-Granger cointegration test result."""
    asset_a: str
    asset_b: str
    cointegrated: bool
    adf_stat: float
    p_value: float
    hedge_ratio: float         # OLS beta
    half_life: float           # mean-reversion half-life in days


@dataclass
class SpreadSignal:
    """Z-score based spread trading signal."""
    asset_a: str
    asset_b: str
    current_spread: float
    z_score: float
    signal: str                # "long_spread", "short_spread", "neutral"
    entry_threshold: float
    exit_threshold: float
    hedge_ratio: float


@dataclass
class MomentumSignal:
    """Cross-asset relative strength signal."""
    asset: str
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    rank: int
    signal: str                # "overweight", "underweight", "neutral"


@dataclass
class MacroRegimeSignal:
    """Macro regime signal from market indicators."""
    indicator: str
    value: float
    z_score: float
    regime: str                # "risk_on", "risk_off", "neutral"
    description: str


@dataclass
class SignalDashboard:
    """Aggregated signal summary."""
    overall_regime: str
    confidence: float
    n_risk_on: int
    n_risk_off: int
    n_neutral: int
    top_signals: List[str]


# ── ADF test (pure numpy/scipy) ────────────────────────────────────────


def _adf_test(series: np.ndarray, max_lags: int = 1) -> Tuple[float, float]:
    """Simplified Augmented Dickey-Fuller test.

    Returns (adf_statistic, approximate_p_value).
    Uses OLS on Δy_t = α + β·y_{t-1} + ε_t.
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 10:
        return 0.0, 1.0

    dy = np.diff(y)
    y_lag = y[:-1]

    # OLS: dy = a + b * y_lag
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    b = beta[1]
    residuals = dy - X @ beta
    se = np.sqrt(np.sum(residuals ** 2) / (len(dy) - 2))
    se_b = se / np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    if se_b < 1e-15:
        return 0.0, 1.0

    adf_stat = b / se_b

    # Approximate p-value using MacKinnon critical values (n→∞)
    # 1%: -3.43, 5%: -2.86, 10%: -2.57
    if adf_stat < -3.43:
        p = 0.005
    elif adf_stat < -2.86:
        p = 0.03
    elif adf_stat < -2.57:
        p = 0.07
    elif adf_stat < -1.94:
        p = 0.15
    else:
        p = min(1.0, 0.5 + 0.3 * (adf_stat + 1.94))

    return float(adf_stat), float(max(0.0, min(1.0, p)))


# ── Signal generator ───────────────────────────────────────────────────


class CrossAssetSignalGenerator:
    """Generate cross-asset trading signals from multi-asset price data."""

    def __init__(
        self,
        prices: pd.DataFrame,
        window: int = 60,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        lead_lag_max: int = 10,
        cointegration_pvalue: float = 0.05,
        macro_indicators: Optional[pd.DataFrame] = None,
    ) -> None:
        self.prices = prices.copy()
        self.assets = list(prices.columns)
        self.returns = prices.pct_change().dropna()
        self.window = window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lead_lag_max = lead_lag_max
        self.cointegration_pvalue = cointegration_pvalue
        self.macro_indicators = macro_indicators

        # Results
        self.correlations: List[InterMarketCorrelation] = []
        self.lead_lags: List[LeadLagResult] = []
        self.cointegrations: List[CointegrationResult] = []
        self.spread_signals: List[SpreadSignal] = []
        self.momentum_signals: List[MomentumSignal] = []
        self.macro_signals: List[MacroRegimeSignal] = []
        self.dashboard: Optional[SignalDashboard] = None

    @classmethod
    def from_csv(
        cls, prices_path: str,
        macro_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "CrossAssetSignalGenerator":
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        macro = None
        if macro_path:
            macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        return cls(prices, macro_indicators=macro, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.correlations = self._inter_market_correlations()
        self.lead_lags = self._lead_lag_detection()
        self.cointegrations = self._cointegration_tests()
        self.spread_signals = self._spread_trading_signals()
        self.momentum_signals = self._cross_asset_momentum()
        self.macro_signals = self._macro_regime_signals()
        self.dashboard = self._build_dashboard()
        return {
            "correlations": self.correlations,
            "lead_lags": self.lead_lags,
            "cointegrations": self.cointegrations,
            "spread_signals": self.spread_signals,
            "momentum_signals": self.momentum_signals,
            "macro_signals": self.macro_signals,
            "dashboard": self.dashboard,
        }

    # ── Inter-market correlations ───────────────────────────────────────

    def _inter_market_correlations(self) -> List[InterMarketCorrelation]:
        results: List[InterMarketCorrelation] = []
        n = len(self.assets)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.assets[i], self.assets[j]
                ra, rb = self.returns[a], self.returns[b]
                full_corr = float(ra.corr(rb))
                rolling = ra.rolling(self.window).corr(rb).dropna()
                if len(rolling) < 2:
                    continue
                mean_rc = float(rolling.mean())
                std_rc = float(rolling.std())
                current = float(rolling.iloc[-1])
                z = (current - mean_rc) / std_rc if std_rc > 1e-10 else 0.0
                results.append(InterMarketCorrelation(
                    asset_a=a, asset_b=b, correlation=full_corr,
                    rolling_corr_mean=mean_rc, rolling_corr_std=std_rc,
                    current_corr=current, z_score=z,
                ))
        return results

    # ── Lead-lag detection ──────────────────────────────────────────────

    def _lead_lag_detection(self) -> List[LeadLagResult]:
        results: List[LeadLagResult] = []
        n = len(self.assets)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.assets[i], self.assets[j]
                ra = self.returns[a].values
                rb = self.returns[b].values
                best_lag, best_corr = self._find_optimal_lag(ra, rb)
                if abs(best_corr) < 0.05:
                    continue
                # Approximate p-value via t-test
                n_obs = len(ra) - abs(best_lag)
                if n_obs < 5:
                    continue
                t_stat = best_corr * np.sqrt(n_obs - 2) / np.sqrt(1 - best_corr ** 2 + 1e-15)
                p_val = float(2 * sp_stats.t.sf(abs(t_stat), n_obs - 2))

                if best_lag > 0:
                    leader, lagger = a, b
                elif best_lag < 0:
                    leader, lagger = b, a
                    best_lag = -best_lag
                else:
                    leader, lagger = a, b

                results.append(LeadLagResult(
                    leader=leader, lagger=lagger,
                    optimal_lag=best_lag,
                    correlation_at_lag=float(best_corr),
                    p_value=p_val,
                    direction="positive" if best_corr > 0 else "negative",
                ))
        return sorted(results, key=lambda r: -abs(r.correlation_at_lag))

    def _find_optimal_lag(self, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find lag that maximizes absolute cross-correlation."""
        best_lag = 0
        best_corr = 0.0
        for lag in range(-self.lead_lag_max, self.lead_lag_max + 1):
            if lag > 0:
                x_s, y_s = x[:-lag], y[lag:]
            elif lag < 0:
                x_s, y_s = x[-lag:], y[:lag]
            else:
                x_s, y_s = x, y
            if len(x_s) < 10:
                continue
            c = float(np.corrcoef(x_s, y_s)[0, 1])
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag
        return best_lag, best_corr

    # ── Cointegration testing ───────────────────────────────────────────

    def _cointegration_tests(self) -> List[CointegrationResult]:
        results: List[CointegrationResult] = []
        n = len(self.assets)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.assets[i], self.assets[j]
                pa = self.prices[a].dropna()
                pb = self.prices[b].dropna()
                common = pa.index.intersection(pb.index)
                if len(common) < 30:
                    continue
                ya = pa.loc[common].values.astype(float)
                yb = pb.loc[common].values.astype(float)

                # Engle-Granger: OLS regression, then ADF on residuals
                X = np.column_stack([np.ones(len(yb)), yb])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X, ya, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                hedge_ratio = float(beta[1])
                residuals = ya - X @ beta

                adf_stat, p_val = _adf_test(residuals)
                cointegrated = p_val < self.cointegration_pvalue

                # Half-life of mean reversion
                half_life = self._mean_reversion_half_life(residuals)

                results.append(CointegrationResult(
                    asset_a=a, asset_b=b,
                    cointegrated=cointegrated,
                    adf_stat=adf_stat, p_value=p_val,
                    hedge_ratio=hedge_ratio,
                    half_life=half_life,
                ))
        return sorted(results, key=lambda r: r.p_value)

    @staticmethod
    def _mean_reversion_half_life(residuals: np.ndarray) -> float:
        """Estimate half-life from AR(1) coefficient on residuals."""
        y = residuals[1:]
        y_lag = residuals[:-1]
        if len(y) < 5 or np.std(y_lag) < 1e-15:
            return float("inf")
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        phi = beta[1]
        if phi >= 1.0 or phi <= 0.0:
            return float("inf")
        return float(-np.log(2) / np.log(phi))

    # ── Spread trading signals ──────────────────────────────────────────

    def _spread_trading_signals(self) -> List[SpreadSignal]:
        results: List[SpreadSignal] = []
        for coint in self.cointegrations:
            if not coint.cointegrated:
                continue
            a, b = coint.asset_a, coint.asset_b
            pa = self.prices[a]
            pb = self.prices[b]
            spread = pa - coint.hedge_ratio * pb
            spread = spread.dropna()
            if len(spread) < self.window:
                continue
            rolling_mean = spread.rolling(self.window).mean()
            rolling_std = spread.rolling(self.window).std()
            z = ((spread - rolling_mean) / rolling_std).dropna()
            if len(z) == 0:
                continue
            current_z = float(z.iloc[-1])
            current_spread = float(spread.iloc[-1])

            if current_z > self.zscore_entry:
                signal = "short_spread"
            elif current_z < -self.zscore_entry:
                signal = "long_spread"
            else:
                signal = "neutral"

            results.append(SpreadSignal(
                asset_a=a, asset_b=b,
                current_spread=current_spread,
                z_score=current_z,
                signal=signal,
                entry_threshold=self.zscore_entry,
                exit_threshold=self.zscore_exit,
                hedge_ratio=coint.hedge_ratio,
            ))
        return sorted(results, key=lambda s: -abs(s.z_score))

    # ── Cross-asset momentum ───────────────────────────────────────────

    def _cross_asset_momentum(self) -> List[MomentumSignal]:
        results: List[MomentumSignal] = []
        for asset in self.assets:
            p = self.prices[asset].dropna()
            if len(p) < 130:
                continue
            m1 = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0.0
            m3 = float(p.iloc[-1] / p.iloc[-63] - 1) if len(p) >= 63 else 0.0
            m6 = float(p.iloc[-1] / p.iloc[-126] - 1) if len(p) >= 126 else 0.0
            results.append(MomentumSignal(
                asset=asset, momentum_1m=m1, momentum_3m=m3,
                momentum_6m=m6, rank=0, signal="neutral",
            ))
        # Rank by composite momentum (equal weight 1m+3m+6m)
        results.sort(key=lambda m: -(m.momentum_1m + m.momentum_3m + m.momentum_6m))
        n = len(results)
        for i, m in enumerate(results):
            m.rank = i + 1
            if i < n * 0.3:
                m.signal = "overweight"
            elif i >= n * 0.7:
                m.signal = "underweight"
            else:
                m.signal = "neutral"
        return results

    # ── Macro regime signals ────────────────────────────────────────────

    def _macro_regime_signals(self) -> List[MacroRegimeSignal]:
        results: List[MacroRegimeSignal] = []
        if self.macro_indicators is None or self.macro_indicators.empty:
            return results
        for col in self.macro_indicators.columns:
            series = self.macro_indicators[col].dropna()
            if len(series) < 20:
                continue
            current = float(series.iloc[-1])
            mean = float(series.mean())
            std = float(series.std())
            z = (current - mean) / std if std > 1e-10 else 0.0

            # Determine regime based on indicator name and z-score
            lower = col.lower()
            if "vix" in lower or "vol" in lower:
                regime = "risk_off" if z > 1.0 else "risk_on" if z < -0.5 else "neutral"
                desc = f"VIX/vol z-score {z:.2f}: {'elevated' if z > 0 else 'subdued'}"
            elif "yield" in lower or "curve" in lower:
                regime = "risk_off" if z < -1.0 else "risk_on" if z > 0.5 else "neutral"
                desc = f"Yield curve z-score {z:.2f}: {'inverted' if z < 0 else 'steepening'}"
            elif "credit" in lower or "spread" in lower:
                regime = "risk_off" if z > 1.0 else "risk_on" if z < -0.5 else "neutral"
                desc = f"Credit spread z-score {z:.2f}: {'widening' if z > 0 else 'tightening'}"
            else:
                regime = "risk_off" if z > 1.5 else "risk_on" if z < -1.5 else "neutral"
                desc = f"{col} z-score {z:.2f}"

            results.append(MacroRegimeSignal(
                indicator=col, value=current, z_score=z,
                regime=regime, description=desc,
            ))
        return results

    # ── Dashboard ───────────────────────────────────────────────────────

    def _build_dashboard(self) -> SignalDashboard:
        all_signals: List[str] = []
        risk_on = risk_off = neutral = 0

        for m in self.macro_signals:
            if m.regime == "risk_on":
                risk_on += 1
            elif m.regime == "risk_off":
                risk_off += 1
            else:
                neutral += 1

        # Spread signals also contribute
        for s in self.spread_signals:
            if s.signal != "neutral":
                all_signals.append(f"{s.asset_a}/{s.asset_b}: {s.signal} (z={s.z_score:.1f})")

        for m in self.momentum_signals[:3]:
            all_signals.append(f"{m.asset}: {m.signal} (rank {m.rank})")

        for m in self.macro_signals:
            if m.regime != "neutral":
                all_signals.append(f"{m.indicator}: {m.regime}")

        if risk_on > risk_off:
            overall = "risk_on"
        elif risk_off > risk_on:
            overall = "risk_off"
        else:
            overall = "neutral"

        total_macro = risk_on + risk_off + neutral
        confidence = abs(risk_on - risk_off) / max(total_macro, 1)

        return SignalDashboard(
            overall_regime=overall,
            confidence=min(confidence, 1.0),
            n_risk_on=risk_on, n_risk_off=risk_off, n_neutral=neutral,
            top_signals=all_signals[:10],
        )

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.dashboard is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    # ── Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["correlation_heatmap"] = self._chart_correlation_heatmap()
        charts["lead_lag_matrix"] = self._chart_lead_lag_matrix()
        charts["spread_zscores"] = self._chart_spread_zscores()
        charts["momentum"] = self._chart_momentum()
        return charts

    def _chart_correlation_heatmap(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        corr = self.returns.corr()
        n = len(corr)
        if n < 2:
            return ""
        fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))
        im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black")
        fig.colorbar(im, shrink=0.8)
        ax.set_title("Inter-Market Correlation", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_lead_lag_matrix(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.lead_lags:
            return ""
        n = len(self.assets)
        matrix = np.zeros((n, n))
        idx = {a: i for i, a in enumerate(self.assets)}
        for ll in self.lead_lags:
            if ll.leader in idx and ll.lagger in idx:
                matrix[idx[ll.leader], idx[ll.lagger]] = ll.optimal_lag
                matrix[idx[ll.lagger], idx[ll.leader]] = -ll.optimal_lag

        fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))
        vmax = max(abs(matrix.max()), abs(matrix.min()), 1)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(self.assets, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(self.assets, fontsize=8)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] != 0:
                    ax.text(j, i, f"{matrix[i, j]:.0f}d",
                            ha="center", va="center", fontsize=7)
        fig.colorbar(im, shrink=0.8, label="Lead (days)")
        ax.set_title("Lead-Lag Matrix (positive = row leads column)", fontsize=10)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_spread_zscores(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.spread_signals:
            return ""
        fig, ax = plt.subplots(figsize=(8, max(3, len(self.spread_signals) * 0.5)))
        labels = [f"{s.asset_a}/{s.asset_b}" for s in self.spread_signals]
        zscores = [s.z_score for s in self.spread_signals]
        colors = ["#dc2626" if abs(z) > self.zscore_entry else "#f59e0b" if abs(z) > 1 else "#16a34a"
                  for z in zscores]
        ax.barh(labels, zscores, color=colors, alpha=0.85)
        ax.axvline(self.zscore_entry, color="#dc2626", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(-self.zscore_entry, color="#dc2626", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Z-Score")
        ax.set_title("Spread Z-Scores", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_momentum(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.momentum_signals:
            return ""
        fig, ax = plt.subplots(figsize=(8, max(3, len(self.momentum_signals) * 0.4)))
        names = [m.asset for m in self.momentum_signals]
        m1 = [m.momentum_1m for m in self.momentum_signals]
        m3 = [m.momentum_3m for m in self.momentum_signals]
        x = np.arange(len(names))
        w = 0.35
        ax.barh(x - w / 2, m1, w, label="1M", color="#3b82f6", alpha=0.85)
        ax.barh(x + w / 2, m3, w, label="3M", color="#f59e0b", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Return")
        ax.set_title("Cross-Asset Momentum", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        db = self.dashboard or SignalDashboard("neutral", 0, 0, 0, 0, [])
        regime_colors = {"risk_on": "#16a34a", "risk_off": "#dc2626", "neutral": "#64748b"}
        rc = regime_colors.get(db.overall_regime, "#64748b")

        # Correlation table
        corr_rows = ""
        for c in sorted(self.correlations, key=lambda x: -abs(x.correlation)):
            z_cls = "bad" if abs(c.z_score) > 2 else ""
            corr_rows += (
                f'<tr><td>{c.asset_a} / {c.asset_b}</td>'
                f'<td>{c.correlation:.3f}</td>'
                f'<td>{c.current_corr:.3f}</td>'
                f'<td class="{z_cls}">{c.z_score:+.2f}</td></tr>\n'
            )

        # Lead-lag table
        ll_rows = ""
        for ll in self.lead_lags[:15]:
            ll_rows += (
                f'<tr><td>{ll.leader}</td><td>{ll.lagger}</td>'
                f'<td>{ll.optimal_lag}d</td>'
                f'<td>{ll.correlation_at_lag:.3f}</td>'
                f'<td>{ll.p_value:.4f}</td>'
                f'<td>{ll.direction}</td></tr>\n'
            )
        if not ll_rows:
            ll_rows = '<tr><td colspan="6" style="text-align:center;color:#64748b">No significant lead-lag</td></tr>'

        # Cointegration table
        coint_rows = ""
        for ci in self.cointegrations:
            cls = "good" if ci.cointegrated else ""
            coint_rows += (
                f'<tr><td>{ci.asset_a} / {ci.asset_b}</td>'
                f'<td class="{cls}">{"Yes" if ci.cointegrated else "No"}</td>'
                f'<td>{ci.adf_stat:.3f}</td>'
                f'<td>{ci.p_value:.4f}</td>'
                f'<td>{ci.hedge_ratio:.3f}</td>'
                f'<td>{ci.half_life:.1f}d</td></tr>\n'
            )

        # Spread signals
        spread_rows = ""
        for s in self.spread_signals:
            sig_cls = "good" if s.signal == "long_spread" else "bad" if s.signal == "short_spread" else ""
            spread_rows += (
                f'<tr><td>{s.asset_a} / {s.asset_b}</td>'
                f'<td>{s.z_score:+.2f}</td>'
                f'<td class="{sig_cls}">{s.signal}</td>'
                f'<td>{s.hedge_ratio:.3f}</td></tr>\n'
            )
        if not spread_rows:
            spread_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No cointegrated pairs</td></tr>'

        # Momentum table
        mom_rows = ""
        for m in self.momentum_signals:
            sig_cls = "good" if m.signal == "overweight" else "bad" if m.signal == "underweight" else ""
            mom_rows += (
                f'<tr><td>{m.asset}</td><td>{m.rank}</td>'
                f'<td>{m.momentum_1m:+.1%}</td>'
                f'<td>{m.momentum_3m:+.1%}</td>'
                f'<td>{m.momentum_6m:+.1%}</td>'
                f'<td class="{sig_cls}">{m.signal}</td></tr>\n'
            )

        # Macro signals
        macro_rows = ""
        for ms in self.macro_signals:
            cls = {"risk_on": "good", "risk_off": "bad"}.get(ms.regime, "")
            macro_rows += (
                f'<tr><td>{ms.indicator}</td>'
                f'<td>{ms.value:.4f}</td>'
                f'<td>{ms.z_score:+.2f}</td>'
                f'<td class="{cls}">{ms.regime}</td>'
                f'<td>{ms.description}</td></tr>\n'
            )
        if not macro_rows:
            macro_rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">No macro indicators</td></tr>'

        # Dashboard signals list
        signal_list = "".join(f"<li>{s}</li>" for s in db.top_signals) or "<li>No active signals</li>"

        n_coint = sum(1 for c in self.cointegrations if c.cointegrated)

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-Asset Signal Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .risk-badge {{ display: inline-block; padding: 0.3em 0.8em; border-radius: 4px;
                 color: white; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  ul {{ margin: 0.5em 0; padding-left: 1.5em; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Cross-Asset Signal Dashboard</h1>
<div class="meta">{len(self.assets)} assets &middot; {len(self.returns)} observations &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value"><span class="risk-badge" style="background:{rc}">{db.overall_regime.upper().replace('_',' ')}</span></div><div class="label">Macro Regime</div></div>
  <div class="kpi"><div class="value">{db.confidence:.0%}</div><div class="label">Confidence</div></div>
  <div class="kpi"><div class="value">{n_coint}</div><div class="label">Cointegrated Pairs</div></div>
  <div class="kpi"><div class="value">{len(self.spread_signals)}</div><div class="label">Active Spreads</div></div>
  <div class="kpi"><div class="value">{len(self.lead_lags)}</div><div class="label">Lead-Lag Pairs</div></div>
</div>

<h2>1. Signal Summary</h2>
<ul>{signal_list}</ul>

<h2>2. Inter-Market Correlations</h2>
{_img("correlation_heatmap")}
<table>
<thead><tr><th>Pair</th><th>Full Corr</th><th>Current</th><th>Z-Score</th></tr></thead>
<tbody>{corr_rows}</tbody>
</table>

<h2>3. Lead-Lag Detection</h2>
{_img("lead_lag_matrix")}
<table>
<thead><tr><th>Leader</th><th>Lagger</th><th>Lag</th><th>Corr</th><th>p-value</th><th>Direction</th></tr></thead>
<tbody>{ll_rows}</tbody>
</table>

<h2>4. Cointegration (Engle-Granger)</h2>
<table>
<thead><tr><th>Pair</th><th>Cointegrated</th><th>ADF Stat</th><th>p-value</th><th>Hedge Ratio</th><th>Half-Life</th></tr></thead>
<tbody>{coint_rows}</tbody>
</table>

<h2>5. Spread Trading Signals</h2>
{_img("spread_zscores")}
<table>
<thead><tr><th>Pair</th><th>Z-Score</th><th>Signal</th><th>Hedge Ratio</th></tr></thead>
<tbody>{spread_rows}</tbody>
</table>

<h2>6. Cross-Asset Momentum</h2>
{_img("momentum")}
<table>
<thead><tr><th>Asset</th><th>Rank</th><th>1M</th><th>3M</th><th>6M</th><th>Signal</th></tr></thead>
<tbody>{mom_rows}</tbody>
</table>

<h2>7. Macro Regime Signals</h2>
<table>
<thead><tr><th>Indicator</th><th>Value</th><th>Z-Score</th><th>Regime</th><th>Description</th></tr></thead>
<tbody>{macro_rows}</tbody>
</table>

<footer>Generated by <code>compass/cross_asset_signals.py</code></footer>
</body></html>"""
        return html
