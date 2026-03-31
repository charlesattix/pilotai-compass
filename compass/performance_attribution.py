"""
Performance attribution engine — decomposes portfolio returns into
actionable sources of alpha and beta.

Components:
  1. Brinson-style attribution   (allocation vs selection effect)
  2. Factor attribution           (market, vol, regime, timing, selection)
  3. Time-series rolling factors  (rolling contribution of each factor)
  4. Per-experiment attribution   (which experiments drove returns)
  5. Skill vs luck analysis       (bootstrap significance of alpha)
  6. HTML report                  (waterfall, timeline, experiment bars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BrinsonAttribution:
    """Brinson-style single-period attribution."""
    date: Optional[datetime] = None
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    total_active: float = 0.0


@dataclass
class FactorDecomposition:
    """Return decomposed into factor contributions."""
    total_return: float = 0.0
    market: float = 0.0
    volatility: float = 0.0
    regime: float = 0.0
    timing: float = 0.0
    selection: float = 0.0
    residual: float = 0.0


@dataclass
class ExperimentContribution:
    """A single experiment's contribution to portfolio return."""
    name: str
    weight: float
    experiment_return: float
    contribution: float      # weight * return
    pct_of_total: float      # contribution / total_portfolio_return


@dataclass
class StrategyDecomposition:
    """P&L attribution for one strategy type."""
    strategy: str
    total_pnl: float
    n_trades: int
    avg_pnl: float
    win_rate: float
    contribution_pct: float  # fraction of total portfolio P&L


@dataclass
class PeriodAttribution:
    """Attribution for a calendar period (month or quarter)."""
    period_label: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_return: float
    n_trades: int
    market_contribution: float
    selection_contribution: float
    residual: float


@dataclass
class SkillTestResult:
    """Bootstrap significance test for alpha."""
    observed_alpha: float
    bootstrap_mean: float
    bootstrap_std: float
    p_value: float
    confidence_level: float
    is_significant: bool
    n_bootstrap: int


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class PerformanceAttribution:
    """Performance attribution engine.

    Args:
        risk_free_rate: Annualised risk-free rate for excess-return calcs.
        bootstrap_n: Number of bootstrap samples for skill test.
        confidence: Confidence level for significance test.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        bootstrap_n: int = 5_000,
        confidence: float = 0.95,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.bootstrap_n = bootstrap_n
        self.confidence = confidence

    # ------------------------------------------------------------------
    # 1. Brinson attribution
    # ------------------------------------------------------------------

    @staticmethod
    def brinson_attribution(
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        date: Optional[datetime] = None,
    ) -> BrinsonAttribution:
        """Single-period Brinson-Fachler attribution.

        Args:
            portfolio_weights: (N,) actual portfolio weights per segment.
            benchmark_weights: (N,) benchmark weights per segment.
            portfolio_returns: (N,) segment returns in portfolio.
            benchmark_returns: (N,) segment returns in benchmark.
        """
        pw = np.asarray(portfolio_weights, dtype=float)
        bw = np.asarray(benchmark_weights, dtype=float)
        pr = np.asarray(portfolio_returns, dtype=float)
        br = np.asarray(benchmark_returns, dtype=float)

        bm_total = float(bw @ br)

        alloc = float(((pw - bw) * (br - bm_total)).sum())
        sel = float((bw * (pr - br)).sum())
        inter = float(((pw - bw) * (pr - br)).sum())

        return BrinsonAttribution(
            date=date,
            allocation_effect=alloc,
            selection_effect=sel,
            interaction_effect=inter,
            total_active=alloc + sel + inter,
        )

    def brinson_series(
        self,
        portfolio_weights_ts: pd.DataFrame,
        benchmark_weights_ts: pd.DataFrame,
        portfolio_returns_ts: pd.DataFrame,
        benchmark_returns_ts: pd.DataFrame,
    ) -> List[BrinsonAttribution]:
        """Multi-period Brinson attribution, one per row."""
        idx = portfolio_weights_ts.index.intersection(benchmark_weights_ts.index)
        results: List[BrinsonAttribution] = []
        for dt in idx:
            ba = self.brinson_attribution(
                portfolio_weights_ts.loc[dt].values,
                benchmark_weights_ts.loc[dt].values,
                portfolio_returns_ts.loc[dt].values,
                benchmark_returns_ts.loc[dt].values,
                date=dt,
            )
            results.append(ba)
        return results

    # ------------------------------------------------------------------
    # 2. Factor attribution
    # ------------------------------------------------------------------

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        vol_factor: Optional[pd.Series] = None,
        regime_factor: Optional[pd.Series] = None,
        timing_factor: Optional[pd.Series] = None,
    ) -> FactorDecomposition:
        """OLS-based multi-factor attribution.

        Regresses portfolio excess returns on supplied factor series.
        Missing factors are skipped; residual captures unmodelled alpha.
        """
        rf_daily = self.risk_free_rate / TRADING_DAYS
        y = portfolio_returns - rf_daily

        factors: Dict[str, pd.Series] = {"market": market_returns - rf_daily}
        if vol_factor is not None:
            factors["volatility"] = vol_factor
        if regime_factor is not None:
            factors["regime"] = regime_factor
        if timing_factor is not None:
            factors["timing"] = timing_factor

        X = pd.DataFrame(factors)
        aligned = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(aligned) < 5:
            return FactorDecomposition(total_return=float(y.sum()))

        ya = aligned["y"].values
        Xa = aligned.drop(columns="y").values
        Xa_c = np.column_stack([np.ones(len(ya)), Xa])  # intercept

        try:
            betas, _, _, _ = np.linalg.lstsq(Xa_c, ya, rcond=None)
        except np.linalg.LinAlgError:
            return FactorDecomposition(total_return=float(y.sum()))

        alpha = betas[0]
        factor_betas = betas[1:]

        cols = list(aligned.drop(columns="y").columns)
        contrib: Dict[str, float] = {}
        for i, col in enumerate(cols):
            contrib[col] = float(factor_betas[i] * aligned[col].sum())

        predicted = Xa_c @ betas
        residual_total = float((ya - predicted).sum())

        return FactorDecomposition(
            total_return=float(ya.sum()),
            market=contrib.get("market", 0.0),
            volatility=contrib.get("volatility", 0.0),
            regime=contrib.get("regime", 0.0),
            timing=contrib.get("timing", 0.0),
            selection=float(alpha * len(ya)),
            residual=residual_total,
        )

    # ------------------------------------------------------------------
    # 3. Rolling factor attribution
    # ------------------------------------------------------------------

    def rolling_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 63,
        vol_factor: Optional[pd.Series] = None,
        regime_factor: Optional[pd.Series] = None,
        timing_factor: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Rolling-window factor decomposition.

        Returns a DataFrame indexed by date with columns:
        market, volatility, regime, timing, selection, residual.
        """
        rf_daily = self.risk_free_rate / TRADING_DAYS
        y = portfolio_returns - rf_daily
        factors: Dict[str, pd.Series] = {"market": market_returns - rf_daily}
        if vol_factor is not None:
            factors["volatility"] = vol_factor
        if regime_factor is not None:
            factors["regime"] = regime_factor
        if timing_factor is not None:
            factors["timing"] = timing_factor

        X = pd.DataFrame(factors)
        aligned = pd.concat([y.rename("y"), X], axis=1).dropna()

        out_cols = ["market", "volatility", "regime", "timing", "selection", "residual"]
        records: List[Dict] = []

        for end in range(window, len(aligned) + 1):
            start = end - window
            chunk = aligned.iloc[start:end]
            ya = chunk["y"].values
            Xa = chunk.drop(columns="y").values
            Xa_c = np.column_stack([np.ones(len(ya)), Xa])
            try:
                betas, _, _, _ = np.linalg.lstsq(Xa_c, ya, rcond=None)
            except np.linalg.LinAlgError:
                betas = np.zeros(Xa_c.shape[1])

            row: Dict[str, float] = {"date": chunk.index[-1]}
            cols = list(chunk.drop(columns="y").columns)
            for i, col in enumerate(cols):
                row[col] = float(betas[1 + i])
            row["selection"] = float(betas[0])
            row.setdefault("market", 0.0)
            row.setdefault("volatility", 0.0)
            row.setdefault("regime", 0.0)
            row.setdefault("timing", 0.0)
            pred = Xa_c @ betas
            row["residual"] = float((ya - pred).sum())
            records.append(row)

        if not records:
            return pd.DataFrame(columns=["date"] + out_cols)
        df = pd.DataFrame(records).set_index("date")
        return df[out_cols]

    # ------------------------------------------------------------------
    # 4. Per-experiment attribution
    # ------------------------------------------------------------------

    @staticmethod
    def experiment_contributions(
        weights: Dict[str, float],
        returns: Dict[str, float],
    ) -> List[ExperimentContribution]:
        """Attribute portfolio return to individual experiments.

        Args:
            weights: {name: portfolio_weight}
            returns: {name: period_return}
        """
        contribs: List[ExperimentContribution] = []
        total = sum(weights.get(n, 0.0) * returns.get(n, 0.0) for n in weights)

        for name in weights:
            w = weights[name]
            r = returns.get(name, 0.0)
            c = w * r
            pct = c / total if total != 0 else 0.0
            contribs.append(ExperimentContribution(
                name=name, weight=w, experiment_return=r,
                contribution=c, pct_of_total=pct,
            ))
        return contribs

    @staticmethod
    def experiment_contributions_ts(
        weights_ts: pd.DataFrame,
        returns_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Time-series of experiment contributions (weight * return)."""
        aligned_w = weights_ts.reindex(returns_ts.index).ffill().fillna(0)
        return aligned_w * returns_ts

    # ------------------------------------------------------------------
    # 5. Per-strategy decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_decomposition(
        trades: pd.DataFrame,
        strategy_col: str = "strategy_type",
        pnl_col: str = "pnl",
    ) -> List[StrategyDecomposition]:
        """Decompose portfolio P&L by strategy type.

        Args:
            trades: DataFrame with at least strategy_type and pnl columns.
            strategy_col: Column name for strategy type.
            pnl_col: Column name for P&L.
        """
        if trades.empty or strategy_col not in trades.columns:
            return []

        total_pnl = trades[pnl_col].sum()
        results: List[StrategyDecomposition] = []

        for strat, grp in trades.groupby(strategy_col):
            strat_pnl = float(grp[pnl_col].sum())
            n = len(grp)
            wins = int((grp[pnl_col] > 0).sum())
            results.append(StrategyDecomposition(
                strategy=str(strat),
                total_pnl=strat_pnl,
                n_trades=n,
                avg_pnl=strat_pnl / n if n > 0 else 0.0,
                win_rate=wins / n if n > 0 else 0.0,
                contribution_pct=strat_pnl / total_pnl if abs(total_pnl) > 1e-12 else 0.0,
            ))

        return sorted(results, key=lambda s: s.total_pnl, reverse=True)

    # ------------------------------------------------------------------
    # 6. Time-period attribution (monthly / quarterly)
    # ------------------------------------------------------------------

    def period_attribution(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        freq: str = "M",
    ) -> List[PeriodAttribution]:
        """Attribute returns by calendar period.

        Args:
            portfolio_returns: Daily return series.
            market_returns: Daily market return series.
            freq: 'M' for monthly, 'Q' for quarterly.
        """
        aligned = pd.concat(
            [portfolio_returns.rename("port"), market_returns.rename("mkt")],
            axis=1,
        ).dropna()

        if aligned.empty:
            return []

        rf_daily = self.risk_free_rate / TRADING_DAYS
        aligned["excess"] = aligned["port"] - rf_daily
        aligned["mkt_excess"] = aligned["mkt"] - rf_daily

        results: List[PeriodAttribution] = []
        for period_key, grp in aligned.groupby(aligned.index.to_period(freq)):
            if len(grp) < 2:
                continue

            total_ret = float(grp["excess"].sum())
            mkt_contrib = float(grp["mkt_excess"].sum())

            # Simple regression for selection alpha within period
            ya = grp["excess"].values
            xa = grp["mkt_excess"].values
            xa_c = np.column_stack([np.ones(len(ya)), xa])
            try:
                betas, _, _, _ = np.linalg.lstsq(xa_c, ya, rcond=None)
                alpha_contrib = float(betas[0] * len(ya))
            except np.linalg.LinAlgError:
                alpha_contrib = 0.0

            residual = total_ret - mkt_contrib - alpha_contrib

            results.append(PeriodAttribution(
                period_label=str(period_key),
                start_date=grp.index[0],
                end_date=grp.index[-1],
                total_return=total_ret,
                n_trades=len(grp),
                market_contribution=mkt_contrib,
                selection_contribution=alpha_contrib,
                residual=residual,
            ))

        return results

    # ------------------------------------------------------------------
    # 7. Skill vs luck (bootstrap)
    # ------------------------------------------------------------------

    def skill_test(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        seed: int = 42,
    ) -> SkillTestResult:
        """Bootstrap test of whether observed alpha is statistically significant.

        Null hypothesis: alpha = 0 (manager has no skill).
        """
        excess = (portfolio_returns - benchmark_returns).dropna()
        if len(excess) < 10:
            return SkillTestResult(
                observed_alpha=0.0, bootstrap_mean=0.0, bootstrap_std=0.0,
                p_value=1.0, confidence_level=self.confidence,
                is_significant=False, n_bootstrap=0,
            )

        observed = float(excess.mean())
        n = len(excess)
        rng = np.random.default_rng(seed)

        # Bootstrap under null: centre excess returns at zero
        centred = excess.values - observed
        boot_means = np.empty(self.bootstrap_n)
        for i in range(self.bootstrap_n):
            sample = rng.choice(centred, size=n, replace=True)
            boot_means[i] = sample.mean()

        boot_mu = float(boot_means.mean())
        boot_std = float(boot_means.std())

        # One-sided p-value: fraction of bootstrap samples >= observed alpha
        p_value = float((boot_means >= observed).sum() / self.bootstrap_n)

        return SkillTestResult(
            observed_alpha=observed,
            bootstrap_mean=boot_mu,
            bootstrap_std=boot_std,
            p_value=p_value,
            confidence_level=self.confidence,
            is_significant=p_value < (1 - self.confidence),
            n_bootstrap=self.bootstrap_n,
        )

    # ------------------------------------------------------------------
    # 6. HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_waterfall(
        decomp: FactorDecomposition,
        width: int = 700, height: int = 320,
    ) -> str:
        """SVG waterfall chart of factor decomposition."""
        items = [
            ("Market", decomp.market),
            ("Volatility", decomp.volatility),
            ("Regime", decomp.regime),
            ("Timing", decomp.timing),
            ("Selection", decomp.selection),
            ("Residual", decomp.residual),
        ]
        items = [(l, v) for l, v in items if v != 0.0]
        if not items:
            return ""

        pad_l, pad_r, pad_t, pad_b = 90, 20, 30, 50
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

        vals = [v for _, v in items]
        cumvals = np.cumsum([0.0] + vals)
        y_min = min(min(cumvals), 0) * 1.15
        y_max = max(max(cumvals), 0) * 1.15
        if y_max == y_min:
            y_max = y_min + 0.01
        bar_w = pw / max(len(items), 1) * 0.7
        gap = pw / max(len(items), 1)

        def ty(v: float) -> float:
            return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        parts.append(
            f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="13" '
            f'font-weight="bold" fill="#1a1a2e">Factor Waterfall</text>'
        )
        # zero line
        zy = ty(0)
        parts.append(
            f'<line x1="{pad_l}" y1="{zy:.0f}" x2="{width - pad_r}" '
            f'y2="{zy:.0f}" stroke="#999" stroke-width="1"/>'
        )

        running = 0.0
        for i, (label, val) in enumerate(items):
            x = pad_l + i * gap + (gap - bar_w) / 2
            top = running + val if val > 0 else running
            bot = running if val > 0 else running + val
            yt = ty(top)
            yb = ty(bot)
            bh = max(abs(yb - yt), 1)
            color = "#27ae60" if val >= 0 else "#e74c3c"
            parts.append(
                f'<rect x="{x:.0f}" y="{min(yt, yb):.0f}" width="{bar_w:.0f}" '
                f'height="{bh:.0f}" fill="{color}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.0f}" y="{min(yt, yb) - 4:.0f}" '
                f'text-anchor="middle" font-size="10" fill="#333">'
                f'{val:+.2%}</text>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.0f}" y="{height - 12:.0f}" '
                f'text-anchor="middle" font-size="10" fill="#666">{label}</text>'
            )
            running += val

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_rolling_timeline(
        rolling_df: pd.DataFrame,
        width: int = 750, height: int = 280,
    ) -> str:
        """Stacked-area-ish line chart of rolling factor betas."""
        if rolling_df.empty:
            return ""
        cols = [c for c in ["market", "volatility", "regime", "timing", "selection"]
                if c in rolling_df.columns]
        if not cols:
            return ""

        palette = {"market": "#2980b9", "volatility": "#e67e22",
                    "regime": "#8e44ad", "timing": "#1abc9c",
                    "selection": "#e74c3c", "residual": "#999"}
        pad_l, pad_r, pad_t, pad_b = 50, 15, 25, 40
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        n = len(rolling_df)

        all_vals = rolling_df[cols].values.flatten()
        y_min = float(np.nanmin(all_vals)) * 1.1
        y_max = float(np.nanmax(all_vals)) * 1.1
        if y_max == y_min:
            y_max = y_min + 0.01

        def tx(i: int) -> float:
            return pad_l + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        parts.append(
            f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
            f'font-weight="bold" fill="#1a1a2e">Rolling Factor Exposures</text>'
        )
        # zero line
        zy = ty(0) if y_min < 0 < y_max else -1
        if zy > 0:
            parts.append(
                f'<line x1="{pad_l}" y1="{zy:.0f}" x2="{width - pad_r}" '
                f'y2="{zy:.0f}" stroke="#ccc" stroke-dasharray="4,3"/>'
            )

        for ci, col in enumerate(cols):
            vals = rolling_df[col].values
            color = palette.get(col, "#999")
            d = " ".join(
                f"{'M' if j == 0 else 'L'}{tx(j):.1f},{ty(float(vals[j])):.1f}"
                for j in range(n) if not np.isnan(vals[j])
            )
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
            # legend
            lx = pad_l + ci * 110
            parts.append(f'<rect x="{lx}" y="{height - 18}" width="10" height="10" fill="{color}"/>')
            parts.append(
                f'<text x="{lx + 14}" y="{height - 9}" font-size="10" '
                f'fill="#333">{col}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_experiment_bars(
        contribs: List[ExperimentContribution],
        width: int = 600, height: int = 0,
    ) -> str:
        """Horizontal stacked-bar of experiment contributions."""
        if not contribs:
            return ""
        bar_h, gap, pad_l = 26, 6, 110
        if height == 0:
            height = len(contribs) * (bar_h + gap) + 20
        bar_area = width - pad_l - 50

        vals = [c.contribution for c in contribs]
        abs_max = max(abs(v) for v in vals) if vals else 1.0
        if abs_max == 0:
            abs_max = 1.0

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        mid_x = pad_l + bar_area / 2
        parts.append(
            f'<line x1="{mid_x:.0f}" y1="0" x2="{mid_x:.0f}" '
            f'y2="{height}" stroke="#ccc" stroke-width="1"/>'
        )
        for i, c in enumerate(contribs):
            y = 8 + i * (bar_h + gap)
            parts.append(
                f'<text x="{pad_l - 6}" y="{y + bar_h * .7:.0f}" '
                f'text-anchor="end" font-size="11" fill="#333">{c.name}</text>'
            )
            bw = abs(c.contribution) / abs_max * (bar_area / 2)
            color = "#27ae60" if c.contribution >= 0 else "#e74c3c"
            bx = mid_x if c.contribution >= 0 else mid_x - bw
            parts.append(
                f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" '
                f'height="{bar_h}" fill="{color}" rx="3"/>'
            )
            parts.append(
                f'<text x="{bx + bw + 4:.0f}" y="{y + bar_h * .7:.0f}" '
                f'font-size="10" fill="#333">{c.contribution:+.2%}</text>'
            )
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _strategy_html(decomp: Optional[List[StrategyDecomposition]]) -> str:
        if not decomp:
            return ""
        rows = []
        for s in decomp:
            rows.append(
                f"<tr><td style='text-align:left'>{s.strategy}</td>"
                f"<td>{s.n_trades}</td><td>${s.total_pnl:,.2f}</td>"
                f"<td>${s.avg_pnl:,.2f}</td><td>{s.win_rate:.1%}</td>"
                f"<td>{s.contribution_pct:.1%}</td></tr>"
            )
        return f"""
<h2>Strategy Decomposition</h2>
<table><tr><th style='text-align:left'>Strategy</th><th>Trades</th>
<th>Total P&L</th><th>Avg P&L</th><th>Win Rate</th><th>Contribution</th></tr>
{''.join(rows)}</table>"""

    @staticmethod
    def _period_html(periods: Optional[List[PeriodAttribution]]) -> str:
        if not periods:
            return ""
        rows = []
        for p in periods:
            rows.append(
                f"<tr><td style='text-align:left'>{p.period_label}</td>"
                f"<td>{p.total_return:+.4%}</td><td>{p.n_trades}</td>"
                f"<td>{p.market_contribution:+.4%}</td>"
                f"<td>{p.selection_contribution:+.4%}</td>"
                f"<td>{p.residual:+.4%}</td></tr>"
            )
        return f"""
<h2>Period Attribution</h2>
<table><tr><th style='text-align:left'>Period</th><th>Return</th><th>Days</th>
<th>Market</th><th>Selection</th><th>Residual</th></tr>
{''.join(rows)}</table>"""

    def generate_report(
        self,
        factor_decomp: FactorDecomposition,
        rolling_df: Optional[pd.DataFrame] = None,
        skill_result: Optional[SkillTestResult] = None,
        experiment_contribs: Optional[List[ExperimentContribution]] = None,
        brinson: Optional[BrinsonAttribution] = None,
        strategy_decomp: Optional[List[StrategyDecomposition]] = None,
        period_attrs: Optional[List[PeriodAttribution]] = None,
        output_path: str = "reports/performance_attribution.html",
    ) -> str:
        """Full HTML attribution report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # --- Waterfall ---
        waterfall_svg = self._svg_waterfall(factor_decomp)

        # --- Rolling ---
        rolling_svg = ""
        if rolling_df is not None and not rolling_df.empty:
            rolling_svg = (
                '<h2>Rolling Factor Exposures</h2>\n'
                + self._svg_rolling_timeline(rolling_df)
            )

        # --- Experiment bars ---
        exp_svg = ""
        exp_table = ""
        if experiment_contribs:
            exp_svg = (
                '<h2>Experiment Contributions</h2>\n'
                + self._svg_experiment_bars(experiment_contribs)
            )
            rows = []
            for c in experiment_contribs:
                rows.append(
                    f"<tr><td>{c.name}</td><td>{c.weight:.2%}</td>"
                    f"<td>{c.experiment_return:.4%}</td>"
                    f"<td>{c.contribution:+.4%}</td>"
                    f"<td>{c.pct_of_total:.1%}</td></tr>"
                )
            exp_table = f"""
<table><tr><th>Experiment</th><th>Weight</th><th>Return</th>
<th>Contribution</th><th>% of Total</th></tr>
{''.join(rows)}</table>"""

        # --- Skill test ---
        skill_html = ""
        if skill_result is not None:
            sig_label = "SIGNIFICANT" if skill_result.is_significant else "NOT significant"
            sig_color = "#27ae60" if skill_result.is_significant else "#e74c3c"
            skill_html = f"""
<h2>Skill vs Luck Analysis</h2>
<div class="summary">
<p><strong>Observed Alpha:</strong> {skill_result.observed_alpha:.4%} daily
   ({skill_result.observed_alpha * TRADING_DAYS:.2%} annualised)</p>
<p><strong>Bootstrap p-value:</strong> {skill_result.p_value:.4f}
   ({skill_result.n_bootstrap:,} samples)</p>
<p><strong>Result:</strong>
   <span style="color:{sig_color};font-weight:bold">{sig_label}</span>
   at {skill_result.confidence_level:.0%} confidence</p>
</div>"""

        # --- Brinson ---
        brinson_html = ""
        if brinson is not None:
            brinson_html = f"""
<h2>Brinson Attribution</h2>
<table class="metrics">
<tr><th>Allocation</th><th>Selection</th><th>Interaction</th><th>Total Active</th></tr>
<tr><td>{brinson.allocation_effect:+.4%}</td>
<td>{brinson.selection_effect:+.4%}</td>
<td>{brinson.interaction_effect:+.4%}</td>
<td>{brinson.total_active:+.4%}</td></tr></table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Performance Attribution</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.metrics {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; font-weight: 600; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Performance Attribution Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Total Return:</strong> {factor_decomp.total_return:.4%}</p>
</div>

<h2>Factor Decomposition</h2>
{waterfall_svg}
<table class="metrics">
<tr><th>Market</th><th>Volatility</th><th>Regime</th><th>Timing</th>
<th>Selection</th><th>Residual</th></tr>
<tr><td>{factor_decomp.market:+.4%}</td>
<td>{factor_decomp.volatility:+.4%}</td>
<td>{factor_decomp.regime:+.4%}</td>
<td>{factor_decomp.timing:+.4%}</td>
<td>{factor_decomp.selection:+.4%}</td>
<td>{factor_decomp.residual:+.4%}</td></tr></table>

{rolling_svg}
{brinson_html}
{exp_svg}
{exp_table}
{skill_html}
{self._strategy_html(strategy_decomp)}
{self._period_html(period_attrs)}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Attribution report -> %s", path)
        return str(path)
