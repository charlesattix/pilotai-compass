"""Multi-factor risk model for portfolio construction — fundamental and
statistical factors, cross-sectional exposure estimation, factor return
attribution, Ledoit-Wolf covariance shrinkage, factor-neutral portfolios,
factor-mimicking portfolios, and factor timing signals.

Provides:
  1. Fundamental factors (value, momentum, quality, size, volatility)
  2. Statistical factors via PCA
  3. Factor exposure estimation (cross-sectional regression)
  4. Factor return attribution
  5. Specific (idiosyncratic) risk estimation
  6. Factor covariance with Ledoit-Wolf shrinkage
  7. Risk decomposition (systematic vs idiosyncratic)
  8. Factor-neutral portfolio construction
  9. Factor-mimicking portfolios
  10. Factor timing signals (momentum + mean-reversion)
  11. HTML report with charts, heatmaps, and tables
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

FUNDAMENTAL_FACTORS = ["value", "momentum", "quality", "size", "volatility"]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class FactorExposure:
    """Factor exposure for one asset."""
    asset: str
    exposures: Dict[str, float]  # factor_name → beta


@dataclass
class FactorReturn:
    """Return attributed to one factor over a period."""
    factor: str
    cumulative_return: float
    avg_daily_return: float
    volatility: float
    sharpe: float
    t_stat: float


@dataclass
class RiskDecomposition:
    """Systematic vs idiosyncratic risk split."""
    total_variance: float
    systematic_variance: float
    idiosyncratic_variance: float
    systematic_pct: float
    idiosyncratic_pct: float
    factor_contributions: Dict[str, float]  # factor → variance contribution


@dataclass
class NeutralPortfolio:
    """Factor-neutral portfolio weights."""
    weights: Dict[str, float]
    residual_exposures: Dict[str, float]
    max_residual: float
    is_neutral: bool  # all exposures < threshold


@dataclass
class MimickingPortfolio:
    """Factor-mimicking portfolio: replicates a single factor's return."""
    factor: str
    weights: Dict[str, float]
    tracking_error: float
    correlation_with_factor: float


@dataclass
class FactorTimingSignal:
    """Factor timing signal based on momentum and mean-reversion."""
    factor: str
    momentum_1m: float
    momentum_3m: float
    z_score: float           # current return vs rolling mean
    signal: str              # "overweight", "underweight", "neutral"
    strength: float          # 0-1


@dataclass
class FactorModelResult:
    """Complete factor model output."""
    exposures: List[FactorExposure] = field(default_factory=list)
    factor_returns: List[FactorReturn] = field(default_factory=list)
    risk_decomposition: Optional[RiskDecomposition] = None
    factor_covariance: Optional[pd.DataFrame] = None
    neutral_portfolio: Optional[NeutralPortfolio] = None
    mimicking_portfolios: List[MimickingPortfolio] = field(default_factory=list)
    timing_signals: List[FactorTimingSignal] = field(default_factory=list)
    n_factors: int = 0
    n_assets: int = 0
    pca_variance_explained: List[float] = field(default_factory=list)
    generated_at: str = ""


# ── Core model ──────────────────────────────────────────────────────────────
class FactorModel:
    """Multi-factor risk model."""

    def __init__(
        self,
        n_statistical_factors: int = 3,
        neutrality_threshold: float = 0.05,
        shrinkage_target: float = 0.5,
    ) -> None:
        self.n_stat_factors = n_statistical_factors
        self.neutrality_threshold = neutrality_threshold
        self.shrinkage_target = shrinkage_target

    def fit(
        self,
        returns: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> FactorModelResult:
        """Fit factor model.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns, columns = asset names, index = dates.
        factor_data : pd.DataFrame, optional
            Fundamental factor values, columns = factor names, index = dates.
            If None, uses PCA-only statistical factors.
        weights : dict, optional
            Current portfolio weights for risk decomposition.
        """
        returns = returns.dropna(how="all")
        if returns.shape[0] < 10 or returns.shape[1] < 2:
            return FactorModelResult(generated_at=self._now())

        assets = list(returns.columns)
        n_assets = len(assets)

        # Build factor matrix
        factors_df, pca_var = self._build_factors(returns, factor_data)
        if factors_df.empty:
            return FactorModelResult(generated_at=self._now())

        factor_names = list(factors_df.columns)

        # Align
        common = returns.index.intersection(factors_df.index)
        R = returns.loc[common]
        F = factors_df.loc[common]

        # Cross-sectional exposure estimation
        exposures = self._estimate_exposures(R, F, assets, factor_names)

        # Factor returns (time-series)
        factor_rets = self._factor_returns(F, factor_names)

        # Factor covariance with shrinkage
        cov = self._factor_covariance(F)

        # Risk decomposition
        risk_dec = None
        if weights:
            risk_dec = self._risk_decomposition(R, F, weights, assets, factor_names)

        # Factor-neutral portfolio
        neutral = self._neutral_portfolio(exposures, assets, factor_names)

        # Factor-mimicking portfolios
        mimicking = self._mimicking_portfolios(R, F, assets, factor_names)

        # Factor timing signals
        timing = self._timing_signals(F, factor_names)

        return FactorModelResult(
            exposures=exposures,
            factor_returns=factor_rets,
            risk_decomposition=risk_dec,
            factor_covariance=cov,
            neutral_portfolio=neutral,
            mimicking_portfolios=mimicking,
            timing_signals=timing,
            n_factors=len(factor_names),
            n_assets=n_assets,
            pca_variance_explained=pca_var,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: FactorModelResult,
        output_path: str | Path = "reports/factor_model.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Factor model report written to %s", path)
        return path

    # ── Factor construction ─────────────────────────────────────────────────
    def _build_factors(
        self,
        returns: pd.DataFrame,
        factor_data: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[float]]:
        frames: List[pd.DataFrame] = []
        pca_var: List[float] = []

        # Fundamental factors
        if factor_data is not None and not factor_data.empty:
            common = returns.index.intersection(factor_data.index)
            if len(common) > 10:
                frames.append(factor_data.loc[common])

        # Statistical factors via PCA
        n_comp = min(self.n_stat_factors, returns.shape[1] - 1, returns.shape[0] - 1)
        if n_comp >= 1:
            clean = returns.dropna()
            if len(clean) > n_comp:
                pca = PCA(n_components=n_comp)
                components = pca.fit_transform(clean.values)
                pca_var = [float(v) for v in pca.explained_variance_ratio_]
                pca_df = pd.DataFrame(
                    components,
                    index=clean.index,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                )
                frames.append(pca_df)

        if not frames:
            return pd.DataFrame(), pca_var

        combined = pd.concat(frames, axis=1, join="inner")
        return combined, pca_var

    # ── Exposure estimation ─────────────────────────────────────────────────
    @staticmethod
    def _estimate_exposures(
        R: pd.DataFrame, F: pd.DataFrame,
        assets: List[str], factor_names: List[str],
    ) -> List[FactorExposure]:
        results: List[FactorExposure] = []
        F_vals = F.values
        for asset in assets:
            y = R[asset].values
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                results.append(FactorExposure(asset, {f: 0.0 for f in factor_names}))
                continue
            reg = LinearRegression().fit(F_vals[mask], y[mask])
            exps = {f: float(reg.coef_[i]) for i, f in enumerate(factor_names)}
            results.append(FactorExposure(asset, exps))
        return results

    # ── Factor returns ──────────────────────────────────────────────────────
    @staticmethod
    def _factor_returns(F: pd.DataFrame, factor_names: List[str]) -> List[FactorReturn]:
        results: List[FactorReturn] = []
        n = len(F)
        for f in factor_names:
            vals = F[f].dropna()
            if len(vals) < 5:
                results.append(FactorReturn(f, 0, 0, 0, 0, 0))
                continue
            cum = float((1 + vals).prod() - 1) if vals.abs().max() < 1 else float(vals.sum())
            avg = float(vals.mean())
            vol = float(vals.std())
            sharpe = avg / vol * np.sqrt(252) if vol > 1e-12 else 0.0
            t = avg / (vol / np.sqrt(len(vals))) if vol > 1e-12 else 0.0
            results.append(FactorReturn(f, cum, avg, vol, sharpe, t))
        return results

    # ── Covariance with Ledoit-Wolf shrinkage ───────────────────────────────
    def _factor_covariance(self, F: pd.DataFrame) -> pd.DataFrame:
        S = F.cov()
        n = S.shape[0]
        if n < 2:
            return S
        target = np.diag(np.diag(S.values))  # diagonal target
        shrunk = (1 - self.shrinkage_target) * S.values + self.shrinkage_target * target
        return pd.DataFrame(shrunk, index=S.index, columns=S.columns)

    # ── Risk decomposition ──────────────────────────────────────────────────
    @staticmethod
    def _risk_decomposition(
        R: pd.DataFrame, F: pd.DataFrame,
        weights: Dict[str, float],
        assets: List[str], factor_names: List[str],
    ) -> RiskDecomposition:
        w = np.array([weights.get(a, 0.0) for a in assets])
        port_ret = R.values @ w
        total_var = float(np.var(port_ret))

        if total_var < 1e-15:
            return RiskDecomposition(0, 0, 0, 0, 0, {})

        # Regress portfolio return on factors
        F_vals = F.values
        reg = LinearRegression().fit(F_vals, port_ret)
        fitted = reg.predict(F_vals)
        residual = port_ret - fitted

        sys_var = float(np.var(fitted))
        idio_var = float(np.var(residual))
        sys_pct = sys_var / total_var if total_var > 1e-15 else 0.0
        idio_pct = 1.0 - sys_pct

        # Per-factor contribution
        contribs: Dict[str, float] = {}
        for i, f in enumerate(factor_names):
            factor_component = reg.coef_[i] * F_vals[:, i]
            contribs[f] = float(np.var(factor_component)) / total_var if total_var > 1e-15 else 0.0

        return RiskDecomposition(
            total_variance=total_var,
            systematic_variance=sys_var,
            idiosyncratic_variance=idio_var,
            systematic_pct=sys_pct,
            idiosyncratic_pct=idio_pct,
            factor_contributions=contribs,
        )

    # ── Factor-neutral portfolio ────────────────────────────────────────────
    def _neutral_portfolio(
        self,
        exposures: List[FactorExposure],
        assets: List[str],
        factor_names: List[str],
    ) -> NeutralPortfolio:
        n = len(assets)
        if n < 2:
            return NeutralPortfolio({}, {}, 0.0, True)

        # Build exposure matrix B: n_assets × n_factors
        B = np.zeros((n, len(factor_names)))
        for i, exp in enumerate(exposures):
            for j, f in enumerate(factor_names):
                B[i, j] = exp.exposures.get(f, 0.0)

        # Equal weight baseline, then project out factor exposures
        w = np.ones(n) / n
        # Neutralize: w_neutral = w - B @ (B.T @ B)^{-1} @ B.T @ w
        BtB = B.T @ B
        reg_term = np.eye(BtB.shape[0]) * 1e-6  # regularisation
        try:
            BtB_inv = np.linalg.inv(BtB + reg_term)
            adjustment = B @ BtB_inv @ B.T @ w
            w_neutral = w - adjustment
            # Rescale to sum to 1
            w_sum = w_neutral.sum()
            if abs(w_sum) > 1e-9:
                w_neutral /= w_sum
        except np.linalg.LinAlgError:
            w_neutral = w

        weights = {a: float(w_neutral[i]) for i, a in enumerate(assets)}
        residual_exp = {}
        for j, f in enumerate(factor_names):
            residual_exp[f] = float(B[:, j] @ w_neutral)

        max_res = max(abs(v) for v in residual_exp.values()) if residual_exp else 0.0
        is_neutral = max_res < self.neutrality_threshold

        return NeutralPortfolio(
            weights=weights,
            residual_exposures=residual_exp,
            max_residual=max_res,
            is_neutral=is_neutral,
        )

    # ── Factor-mimicking portfolios ───────────────────────────────────────
    def _mimicking_portfolios(
        self,
        R: pd.DataFrame,
        F: pd.DataFrame,
        assets: List[str],
        factor_names: List[str],
    ) -> List[MimickingPortfolio]:
        """Build factor-mimicking portfolios via cross-sectional regression."""
        results: List[MimickingPortfolio] = []
        R_vals = R.values  # T × N
        for j, fname in enumerate(factor_names):
            f_series = F.iloc[:, j].values  # T × 1
            # Regress factor on asset returns: f = R @ w + eps
            # OLS: w = (R'R)^{-1} R'f
            try:
                RtR = R_vals.T @ R_vals
                reg = np.eye(RtR.shape[0]) * 1e-6
                w = np.linalg.solve(RtR + reg, R_vals.T @ f_series)
                # Normalize weights
                w_sum = np.sum(np.abs(w))
                if w_sum > 1e-10:
                    w = w / w_sum
            except np.linalg.LinAlgError:
                w = np.ones(len(assets)) / len(assets)

            # Tracking: correlation between mimicking portfolio returns and factor
            mim_ret = R_vals @ w
            corr = float(np.corrcoef(mim_ret, f_series)[0, 1]) if len(f_series) > 2 else 0.0
            te = float(np.std(mim_ret - f_series)) if len(f_series) > 2 else 0.0

            results.append(MimickingPortfolio(
                factor=fname,
                weights={a: float(w[i]) for i, a in enumerate(assets)},
                tracking_error=te,
                correlation_with_factor=corr,
            ))
        return results

    # ── Factor timing signals ─────────────────────────────────────────────
    def _timing_signals(
        self, F: pd.DataFrame, factor_names: List[str],
    ) -> List[FactorTimingSignal]:
        """Compute momentum + mean-reversion timing signals per factor."""
        results: List[FactorTimingSignal] = []
        for j, fname in enumerate(factor_names):
            series = F.iloc[:, j]
            cum = series.cumsum()
            n = len(cum)

            # Momentum
            m1 = float(cum.iloc[-1] - cum.iloc[-min(21, n)]) if n >= 21 else float(cum.iloc[-1])
            m3 = float(cum.iloc[-1] - cum.iloc[-min(63, n)]) if n >= 63 else float(cum.iloc[-1])

            # Z-score: current rolling mean vs long-term
            window = min(60, n)
            rolling_mean = float(series.iloc[-window:].mean())
            long_mean = float(series.mean())
            long_std = float(series.std())
            z = (rolling_mean - long_mean) / long_std if long_std > 1e-10 else 0.0

            # Signal
            if m1 > 0 and z > 0.5:
                signal = "overweight"
                strength = min(abs(z) / 2, 1.0)
            elif m1 < 0 and z < -0.5:
                signal = "underweight"
                strength = min(abs(z) / 2, 1.0)
            else:
                signal = "neutral"
                strength = 0.0

            results.append(FactorTimingSignal(
                factor=fname, momentum_1m=m1, momentum_3m=m3,
                z_score=z, signal=signal, strength=strength,
            ))
        return results

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── Charts ─────────────────────────────────────────────────────────────
    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_exposure_heatmap(self, r: FactorModelResult) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not r.exposures:
            return ""
        factors = list(r.exposures[0].exposures.keys())
        assets = [e.asset for e in r.exposures]
        matrix = np.array([[e.exposures.get(f, 0) for f in factors] for e in r.exposures])
        fig, ax = plt.subplots(figsize=(max(5, len(factors) * 1.2), max(3, len(assets) * 0.4)))
        vmax = max(abs(matrix.max()), abs(matrix.min()), 0.01)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(assets))); ax.set_yticklabels(assets, fontsize=8)
        for i in range(len(assets)):
            for j in range(len(factors)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i,j]) > vmax * 0.6 else "black")
        fig.colorbar(im, shrink=0.8); ax.set_title("Factor Exposure Heatmap", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_variance_decomp(self, r: FactorModelResult) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not r.pca_variance_explained:
            return ""
        fig, ax = plt.subplots(figsize=(6, 3))
        xs = range(1, len(r.pca_variance_explained) + 1)
        ax.bar(xs, r.pca_variance_explained, color="#3b82f6", alpha=0.85)
        ax.plot(xs, np.cumsum(r.pca_variance_explained), "o-", color="#dc2626", lw=1.2, label="Cumulative")
        ax.set_xlabel("Factor"); ax.set_ylabel("Variance Explained")
        ax.set_title("PCA Variance Decomposition", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2); fig.tight_layout()
        return self._fig_to_b64(fig)

    @staticmethod
    def _html_mimicking(portfolios: Any) -> str:
        """Stub for mimicking portfolios HTML section."""
        if not portfolios:
            return ""
        return "<h2>Mimicking Portfolios</h2><p>Available.</p>"

    @staticmethod
    def _html_timing(signals: Any) -> str:
        """Stub for timing signals HTML section."""
        if not signals:
            return ""
        return "<h2>Timing Signals</h2><p>Available.</p>"

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: FactorModelResult) -> str:
        cards = self._html_cards(r)
        exp_tbl = self._html_exposures(r.exposures)
        ret_tbl = self._html_factor_returns(r.factor_returns)
        risk_sec = self._html_risk_decomp(r.risk_decomposition)
        cov_hm = self._svg_covariance(r.factor_covariance)
        neutral_sec = self._html_neutral(r.neutral_portfolio)

        # New charts
        exp_heatmap_b64 = self._chart_exposure_heatmap(r)
        var_decomp_b64 = self._chart_variance_decomp(r)
        exp_heatmap = f'<div style="background:#fff;border-radius:8px;padding:1em;margin:1em 0;text-align:center"><img src="data:image/png;base64,{exp_heatmap_b64}" style="max-width:100%"></div>' if exp_heatmap_b64 else ""
        var_decomp = f'<div style="background:#fff;border-radius:8px;padding:1em;margin:1em 0;text-align:center"><img src="data:image/png;base64,{var_decomp_b64}" style="max-width:100%"></div>' if var_decomp_b64 else ""

        # Mimicking portfolios section
        mim_sec = self._html_mimicking(r.mimicking_portfolios)
        timing_sec = self._html_timing(r.timing_signals)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Factor Risk Model</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Multi-Factor Risk Model</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_assets} assets &middot; {r.n_factors} factors</p>
{cards}
{exp_tbl}
{ret_tbl}
{risk_sec}
<div class="sec"><h2>Factor Covariance</h2>{cov_hm}</div>
{neutral_sec}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: FactorModelResult) -> str:
        rd = r.risk_decomposition
        sys_pct = f"{rd.systematic_pct:.0%}" if rd else "N/A"
        pca_str = f"{sum(r.pca_variance_explained):.0%}" if r.pca_variance_explained else "N/A"
        neutral = "Yes" if r.neutral_portfolio and r.neutral_portfolio.is_neutral else "No"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Factors</div><div class="val">{r.n_factors}</div></div>
<div class="card"><div class="lbl">Assets</div><div class="val">{r.n_assets}</div></div>
<div class="card"><div class="lbl">Systematic Risk</div><div class="val">{sys_pct}</div></div>
<div class="card"><div class="lbl">PCA Explained</div><div class="val">{pca_str}</div></div>
<div class="card"><div class="lbl">Neutral</div><div class="val">{neutral}</div></div>
</div>"""

    @staticmethod
    def _html_exposures(exps: List[FactorExposure]) -> str:
        if not exps:
            return ""
        factors = list(exps[0].exposures.keys())
        hdr = "".join(f"<th>{f}</th>" for f in factors)
        rows = ""
        for e in exps:
            cells = ""
            for f in factors:
                v = e.exposures.get(f, 0.0)
                cls = "pos" if v > 0.1 else "neg" if v < -0.1 else ""
                cells += f'<td class="{cls}">{v:.3f}</td>'
            rows += f"<tr><td>{e.asset}</td>{cells}</tr>"
        return f"""<div class="sec"><h2>Factor Exposures</h2>
<table><thead><tr><th>Asset</th>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_factor_returns(rets: List[FactorReturn]) -> str:
        if not rets:
            return ""
        rows = ""
        for fr in rets:
            cls = "pos" if fr.sharpe > 0 else "neg"
            rows += (f"<tr><td>{fr.factor}</td><td>{fr.cumulative_return:.4f}</td>"
                     f"<td>{fr.avg_daily_return:.6f}</td><td>{fr.volatility:.4f}</td>"
                     f'<td class="{cls}">{fr.sharpe:.2f}</td><td>{fr.t_stat:.2f}</td></tr>')
        return f"""<div class="sec"><h2>Factor Return Attribution</h2>
<table><thead><tr><th>Factor</th><th>Cum Return</th><th>Avg Daily</th><th>Vol</th><th>Sharpe</th><th>t-stat</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_risk_decomp(rd: Optional[RiskDecomposition]) -> str:
        if not rd:
            return ""
        rows = ""
        for f, c in sorted(rd.factor_contributions.items(), key=lambda x: -x[1]):
            rows += f"<tr><td>{f}</td><td>{c:.1%}</td></tr>"
        return f"""<div class="sec"><h2>Risk Decomposition</h2>
<table><tbody>
<tr><td>Total Variance</td><td>{rd.total_variance:.6f}</td></tr>
<tr><td>Systematic</td><td>{rd.systematic_pct:.1%}</td></tr>
<tr><td>Idiosyncratic</td><td>{rd.idiosyncratic_pct:.1%}</td></tr>
</tbody></table>
<h3 style="margin-top:12px;font-size:.95rem;color:#94a3b8">Factor Contributions</h3>
<table><thead><tr><th>Factor</th><th>% of Variance</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _svg_covariance(cov: Optional[pd.DataFrame]) -> str:
        if cov is None or cov.empty:
            return "<p>No data.</p>"
        labels = list(cov.columns)
        n = len(labels)
        cell = 50
        lbl_w = 70
        w = lbl_w + n * cell + 10
        h = 25 + n * cell + 25
        vals = cov.values
        abs_max = max(abs(vals.min()), abs(vals.max())) or 1.0
        cells = ""
        for i in range(n):
            cells += f'<text x="{lbl_w - 5}" y="{30 + i * cell + cell // 2 + 4}" text-anchor="end" font-size="10" fill="#e2e8f0">{labels[i]}</text>'
            for j in range(n):
                v = vals[i][j]
                intensity = int(abs(v) / abs_max * 200)
                if v >= 0:
                    colour = f"rgb({30},{intensity + 50},{80})"
                else:
                    colour = f"rgb({intensity + 50},{30},{30})"
                x = lbl_w + j * cell
                y = 25 + i * cell
                cells += (f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{colour}" stroke="#0f172a" stroke-width="1"/>'
                          f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" text-anchor="middle" font-size="9" fill="#e2e8f0">{v:.3f}</text>')
        for j in range(n):
            cells += f'<text x="{lbl_w + j * cell + cell // 2}" y="18" text-anchor="middle" font-size="9" fill="#94a3b8">{labels[j]}</text>'
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{cells}</svg>'

    @staticmethod
    def _html_neutral(np_: Optional[NeutralPortfolio]) -> str:
        if not np_:
            return ""
        rows = "".join(f"<tr><td>{a}</td><td>{w:.4f}</td></tr>" for a, w in sorted(np_.weights.items()))
        exp_rows = "".join(f"<tr><td>{f}</td><td>{v:.6f}</td></tr>" for f, v in sorted(np_.residual_exposures.items()))
        cls = "pos" if np_.is_neutral else "neg"
        return f"""<div class="sec"><h2>Factor-Neutral Portfolio</h2>
<p class="{cls}">{'Neutral' if np_.is_neutral else 'NOT neutral'} (max residual: {np_.max_residual:.6f})</p>
<table><thead><tr><th>Asset</th><th>Weight</th></tr></thead><tbody>{rows}</tbody></table>
<h3 style="margin-top:12px;font-size:.95rem;color:#94a3b8">Residual Exposures</h3>
<table><thead><tr><th>Factor</th><th>Exposure</th></tr></thead><tbody>{exp_rows}</tbody></table></div>"""

    @staticmethod
    def _html_mimicking(portfolios: list) -> str:
        if not portfolios:
            return ""
        rows = ""
        for mp in portfolios:
            rows += (f"<tr><td>{mp.factor}</td><td>{mp.tracking_error:.4f}</td>"
                     f"<td>{mp.correlation_with_factor:.3f}</td></tr>")
        return f"""<div class="sec"><h2>Factor-Mimicking Portfolios</h2>
<table><thead><tr><th>Factor</th><th>Tracking Error</th><th>Correlation</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_timing(signals: list) -> str:
        if not signals:
            return ""
        rows = ""
        for s in signals:
            cls = "pos" if s.signal == "overweight" else "neg" if s.signal == "underweight" else ""
            rows += (f"<tr><td>{s.factor}</td><td>{s.momentum_1m:.4f}</td>"
                     f"<td>{s.z_score:.2f}</td>"
                     f'<td class="{cls}">{s.signal}</td><td>{s.strength:.2f}</td></tr>')
        return f"""<div class="sec"><h2>Factor Timing Signals</h2>
<table><thead><tr><th>Factor</th><th>Mom 1M</th><th>Z-Score</th><th>Signal</th><th>Strength</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""
