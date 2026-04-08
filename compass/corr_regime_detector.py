"""
Correlation regime detector — early warning via absorption ratio.

Multi-window (20d/60d/120d) rolling correlation matrices, eigenvalue
absorption ratio (Kritzman 2011), correlation regime classification
(normal/breakdown/crisis), early warning signals, and EXP-880 timing
overlay backtest.

Usage::

    from compass.corr_regime_detector import CorrRegimeDetector
    det = CorrRegimeDetector(returns_df)
    results = det.analyze()
    bt = det.backtest_overlay(trade_pnls, trade_dates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────

WINDOWS = {"short": 20, "medium": 60, "long": 120}


@dataclass
class DetectorConfig:
    windows: Dict[str, int] = field(default_factory=lambda: dict(WINDOWS))
    n_eigenvalues: int = 3           # top-N for absorption ratio
    ar_warning_z: float = 1.5        # z-score for warning
    ar_crisis_z: float = 2.5         # z-score for crisis
    dispersion_warning_z: float = 1.5
    lookback_for_z: int = 120        # rolling lookback for z-scores
    early_warning_lead: int = 5      # days before VIX spike to flag
    overlay_scale_warning: float = 0.5
    overlay_scale_crisis: float = 0.2


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class AbsorptionRatio:
    """Kritzman (2011) absorption ratio at one point."""
    date: str
    window: str
    ratio: float               # top-N eigenvalues / total variance
    z_score: float             # vs rolling history
    eigenvalues: List[float]   # all eigenvalues, descending


@dataclass
class CorrDispersion:
    """Correlation dispersion: std of off-diagonal correlations."""
    date: str
    window: str
    dispersion: float
    z_score: float
    mean_corr: float
    max_corr: float
    min_corr: float


@dataclass
class CorrRegime:
    """Classified correlation regime at one point."""
    date: str
    regime: str                # "normal", "breakdown", "crisis"
    ar_z: float                # absorption ratio z-score
    disp_z: float              # dispersion z-score
    confidence: float          # 0-1
    trigger: str               # what triggered the classification


@dataclass
class EarlyWarning:
    """Early warning signal."""
    date: str
    signal: str                # "warning" or "crisis_imminent"
    ar_z: float
    disp_z: float
    lead_days: int             # how many days before actual event
    description: str


@dataclass
class OverlayResult:
    """Backtest result of using correlation detector as EXP-880 overlay."""
    base_pnl: float
    overlay_pnl: float
    base_dd: float
    overlay_dd: float
    dd_improvement_pct: float
    base_sharpe: float
    overlay_sharpe: float
    n_warnings_fired: int
    n_crisis_fired: int
    avg_lead_days: float
    hit_rate: float            # fraction of warnings that preceded drawdown


@dataclass
class AnalysisResult:
    """Full analysis output."""
    absorption_ratios: Dict[str, List[AbsorptionRatio]]
    dispersions: Dict[str, List[CorrDispersion]]
    regimes: List[CorrRegime]
    early_warnings: List[EarlyWarning]
    n_normal: int
    n_breakdown: int
    n_crisis: int
    avg_ar: float
    max_ar: float


# ── Core computations ───────────────────────────────────────────────────


def rolling_correlation_matrix(
    returns: pd.DataFrame, window: int,
) -> List[Tuple[int, np.ndarray]]:
    """Compute rolling correlation matrices.

    Returns list of (index, corr_matrix) tuples.
    """
    n = len(returns)
    results = []
    for i in range(window, n):
        subset = returns.iloc[i - window:i]
        corr = subset.corr().values
        if not np.any(np.isnan(corr)):
            results.append((i, corr))
    return results


def absorption_ratio(corr_matrix: np.ndarray, n_top: int = 3) -> Tuple[float, List[float]]:
    """Compute absorption ratio: sum(top-N eigenvalues) / sum(all eigenvalues).

    Kritzman et al. (2011): higher AR = more systemic risk.
    """
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.sort(eigvals)[::-1]  # descending
    eigvals = np.maximum(eigvals, 0)  # clip negative
    total = eigvals.sum()
    if total < 1e-15:
        return 0.0, []
    n_top = min(n_top, len(eigvals))
    ar = float(eigvals[:n_top].sum() / total)
    return ar, [float(e) for e in eigvals]


def correlation_dispersion(corr_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Std of off-diagonal correlations.  Returns (std, mean, max, min)."""
    n = corr_matrix.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr_matrix[mask]
    return (
        float(np.std(off_diag)),
        float(np.mean(off_diag)),
        float(np.max(off_diag)),
        float(np.min(off_diag)),
    )


def classify_regime(
    ar_z: float, disp_z: float,
    ar_crisis_z: float = 2.5, ar_warning_z: float = 1.5,
    disp_warning_z: float = 1.5,
) -> Tuple[str, float, str]:
    """Classify correlation regime.  Returns (regime, confidence, trigger)."""
    if ar_z > ar_crisis_z:
        return "crisis", min(ar_z / 4, 1.0), f"AR z={ar_z:.1f} > {ar_crisis_z}"
    if ar_z > ar_warning_z or disp_z > disp_warning_z:
        conf = max(ar_z, disp_z) / 3
        triggers = []
        if ar_z > ar_warning_z:
            triggers.append(f"AR z={ar_z:.1f}")
        if disp_z > disp_warning_z:
            triggers.append(f"Disp z={disp_z:.1f}")
        return "breakdown", min(conf, 1.0), " + ".join(triggers)
    return "normal", 0.0, ""


# ── Detector ────────────────────────────────────────────────────────────


class CorrRegimeDetector:
    """Correlation regime detector with early warning system."""

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        self.returns = returns.copy()
        self.config = config or DetectorConfig()
        self.assets = list(returns.columns)
        self.n = len(returns)
        self.result: Optional[AnalysisResult] = None

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "CorrRegimeDetector":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    def analyze(self) -> AnalysisResult:
        """Full correlation regime analysis."""
        cfg = self.config
        all_ars: Dict[str, List[AbsorptionRatio]] = {}
        all_disps: Dict[str, List[CorrDispersion]] = {}

        # Compute AR and dispersion for each window
        for wname, wsize in cfg.windows.items():
            matrices = rolling_correlation_matrix(self.returns, wsize)
            ars: List[AbsorptionRatio] = []
            disps: List[CorrDispersion] = []

            ar_history: List[float] = []
            disp_history: List[float] = []

            for idx, corr in matrices:
                ar_val, eigvals = absorption_ratio(corr, cfg.n_eigenvalues)
                d_std, d_mean, d_max, d_min = correlation_dispersion(corr)

                ar_history.append(ar_val)
                disp_history.append(d_std)

                # Z-scores
                lb = cfg.lookback_for_z
                ar_window = ar_history[-lb:]
                disp_window = disp_history[-lb:]

                ar_z = (ar_val - np.mean(ar_window)) / max(np.std(ar_window), 1e-10) if len(ar_window) > 10 else 0
                disp_z = (d_std - np.mean(disp_window)) / max(np.std(disp_window), 1e-10) if len(disp_window) > 10 else 0

                date = str(self.returns.index[idx]) if hasattr(self.returns.index, '__getitem__') else str(idx)

                ars.append(AbsorptionRatio(date, wname, ar_val, float(ar_z), eigvals))
                disps.append(CorrDispersion(date, wname, d_std, float(disp_z), d_mean, d_max, d_min))

            all_ars[wname] = ars
            all_disps[wname] = disps

        # Classify regimes using the medium window
        regimes: List[CorrRegime] = []
        med_ars = all_ars.get("medium", [])
        med_disps = all_disps.get("medium", [])

        for ar, disp in zip(med_ars, med_disps):
            regime, conf, trigger = classify_regime(
                ar.z_score, disp.z_score,
                cfg.ar_crisis_z, cfg.ar_warning_z, cfg.dispersion_warning_z,
            )
            regimes.append(CorrRegime(ar.date, regime, ar.z_score, disp.z_score, conf, trigger))

        # Early warnings: detect where regime changes from normal
        warnings: List[EarlyWarning] = []
        for i, reg in enumerate(regimes):
            if reg.regime in ("breakdown", "crisis") and (i == 0 or regimes[i - 1].regime == "normal"):
                sig = "crisis_imminent" if reg.regime == "crisis" else "warning"
                warnings.append(EarlyWarning(
                    reg.date, sig, reg.ar_z, reg.disp_z,
                    cfg.early_warning_lead,
                    f"Correlation {reg.regime}: AR z={reg.ar_z:.1f}, Disp z={reg.disp_z:.1f}",
                ))

        n_normal = sum(1 for r in regimes if r.regime == "normal")
        n_breakdown = sum(1 for r in regimes if r.regime == "breakdown")
        n_crisis = sum(1 for r in regimes if r.regime == "crisis")
        avg_ar = float(np.mean([a.ratio for a in med_ars])) if med_ars else 0
        max_ar = float(np.max([a.ratio for a in med_ars])) if med_ars else 0

        self.result = AnalysisResult(
            all_ars, all_disps, regimes, warnings,
            n_normal, n_breakdown, n_crisis, avg_ar, max_ar,
        )
        return self.result

    def backtest_overlay(
        self,
        trade_pnls: np.ndarray,
        trade_date_indices: np.ndarray,
        capital: float = 100_000,
    ) -> OverlayResult:
        """Backtest as EXP-880 timing overlay.

        Reduce position sizing when correlation breakdown detected.
        """
        if self.result is None:
            self.analyze()

        cfg = self.config
        regimes = self.result.regimes

        # Build regime lookup by index
        regime_map: Dict[int, str] = {}
        med_window = cfg.windows.get("medium", 60)
        for i, reg in enumerate(regimes):
            regime_map[i + med_window] = reg.regime

        # Apply overlay
        base_pnls = trade_pnls.copy()
        overlay_pnls = trade_pnls.copy()

        n_warnings = 0
        n_crisis = 0

        for j, idx in enumerate(trade_date_indices):
            idx_int = int(idx)
            regime = regime_map.get(idx_int, "normal")
            # Also check nearby (lead days)
            for offset in range(cfg.early_warning_lead):
                nearby = regime_map.get(idx_int - offset, "normal")
                if nearby in ("breakdown", "crisis"):
                    regime = nearby
                    break

            if regime == "crisis":
                overlay_pnls[j] *= cfg.overlay_scale_crisis
                n_crisis += 1
            elif regime == "breakdown":
                overlay_pnls[j] *= cfg.overlay_scale_warning
                n_warnings += 1

        # Metrics
        base_eq = capital + np.cumsum(base_pnls)
        overlay_eq = capital + np.cumsum(overlay_pnls)

        def _dd(eq):
            full = np.concatenate([[capital], eq])
            pk = np.maximum.accumulate(full)
            return float(np.min((full - pk) / np.where(pk > 0, pk, 1)))

        def _sharpe(pnls):
            rets = pnls / capital
            return float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0

        b_dd = _dd(base_eq)
        o_dd = _dd(overlay_eq)
        dd_imp = (abs(b_dd) - abs(o_dd)) / abs(b_dd) * 100 if abs(b_dd) > 0 else 0

        # Hit rate: what fraction of warnings preceded a loss
        n_hits = 0
        for j, idx in enumerate(trade_date_indices):
            idx_int = int(idx)
            regime = regime_map.get(idx_int, "normal")
            if regime in ("breakdown", "crisis") and base_pnls[j] < 0:
                n_hits += 1
        total_flags = n_warnings + n_crisis
        hit_rate = n_hits / total_flags if total_flags > 0 else 0

        # Avg lead: use early_warning_lead as configured
        avg_lead = float(cfg.early_warning_lead)

        return OverlayResult(
            base_pnl=float(base_pnls.sum()),
            overlay_pnl=float(overlay_pnls.sum()),
            base_dd=b_dd, overlay_dd=o_dd,
            dd_improvement_pct=dd_imp,
            base_sharpe=_sharpe(base_pnls),
            overlay_sharpe=_sharpe(overlay_pnls),
            n_warnings_fired=n_warnings,
            n_crisis_fired=n_crisis,
            avg_lead_days=avg_lead,
            hit_rate=hit_rate,
        )

    def get_current_regime(self) -> Optional[CorrRegime]:
        """Get the most recent correlation regime."""
        if self.result and self.result.regimes:
            return self.result.regimes[-1]
        return None
