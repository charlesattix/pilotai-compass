"""
Implied volatility surface modeler.

Components:
  - SVI parameterization (raw SVI: a, b, rho, m, sigma)
  - Smile interpolation across strikes and expiries
  - No-arbitrage constraints: butterfly spread (convexity), calendar spread
  - Surface dynamics: sticky-strike vs sticky-delta regime detection
  - Implied vol forecasting from surface shape
  - Skew / kurtosis extraction from smile curvature
  - HTML report: 3D surface plot, term structure, skew chart, smile evolution

All methods work on pre-loaded data — no network calls.

Usage::

    from compass.vol_surface import VolSurfaceModeler
    modeler = VolSurfaceModeler()
    result = modeler.analyze(chain_df, forward=450.0)
    VolSurfaceModeler.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "vol_surface.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SVIParams:
    """Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""

    a: float       # overall variance level
    b: float       # slope / tightness
    rho: float     # asymmetry  [-1, 1]
    m: float       # horizontal shift
    sigma: float   # smoothness (ATM curvature)
    expiry: float  # years


@dataclass
class SkewKurtosis:
    """Skew and kurtosis extracted from smile shape."""

    expiry_days: int
    atm_vol: float
    skew_25d: float           # put_vol - call_vol at 25-delta
    butterfly_25d: float      # (put + call)/2 - atm
    implied_skewness: float   # third moment proxy
    implied_kurtosis: float   # fourth moment proxy
    skew_slope: float         # dIV/dlog(K/F) at ATM


@dataclass
class ArbitrageCheck:
    """No-arbitrage constraint check result."""

    violation_type: str     # "butterfly", "calendar"
    strike: float
    expiry_1: float
    expiry_2: Optional[float] = None
    detail: str = ""


@dataclass
class TermStructurePoint:
    """Single ATM term structure point."""

    expiry_days: int
    atm_vol: float
    total_variance: float


@dataclass
class SurfaceDynamics:
    """Sticky-strike vs sticky-delta regime detection."""

    regime: str              # "sticky_strike", "sticky_delta", "mixed"
    sticky_strike_score: float   # correlation of IV changes vs strike
    sticky_delta_score: float    # correlation of IV changes vs moneyness
    confidence: float


@dataclass
class VolForecast:
    """Implied vol forecast from surface shape."""

    current_atm_vol: float
    forecast_atm_vol: float
    term_structure_slope: float   # positive = contango
    skew_signal: float            # negative skew increasing = risk
    confidence: float


@dataclass
class SurfaceResult:
    """Full result from vol surface analysis."""

    surface: pd.DataFrame          # strike x expiry grid
    svi_params_by_expiry: Dict[int, SVIParams]
    skew_kurtosis: List[SkewKurtosis]
    term_structure: List[TermStructurePoint]
    is_contango: bool
    arbitrage_checks: List[ArbitrageCheck]
    dynamics: Optional[SurfaceDynamics]
    forecast: Optional[VolForecast]
    n_strikes: int
    n_expiries: int


# ── SVI model ────────────────────────────────────────────────────────────


def svi_total_variance(k: float, a: float, b: float, rho: float,
                        m: float, sigma: float) -> float:
    """Raw SVI total implied variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k: float, expiry: float, a: float, b: float,
                     rho: float, m: float, sigma: float) -> float:
    """Implied vol from SVI total variance: iv = sqrt(w/T)."""
    if expiry <= 0:
        return 0.0
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = max(w, 1e-12)
    return math.sqrt(w / expiry)


def calibrate_svi(
    log_strikes: np.ndarray,
    market_vols: np.ndarray,
    expiry: float,
) -> SVIParams:
    """Calibrate raw SVI parameters to market vols for one expiry.

    Uses grid search + refinement since scipy.optimize may not be available
    with all solvers. Simple but robust.
    """
    if len(log_strikes) < 3 or expiry <= 0:
        atm_vol = float(market_vols.mean()) if len(market_vols) > 0 else 0.2
        return SVIParams(a=atm_vol ** 2 * expiry, b=0.1, rho=-0.3,
                          m=0.0, sigma=0.1, expiry=expiry)

    total_var = market_vols ** 2 * expiry

    best_err = float("inf")
    best_params = (total_var.mean(), 0.1, -0.3, 0.0, 0.1)

    # Grid search
    for a in np.linspace(total_var.min() * 0.5, total_var.max(), 5):
        for b in [0.05, 0.1, 0.2, 0.4]:
            for rho in [-0.7, -0.3, 0.0, 0.3]:
                for sigma_val in [0.05, 0.1, 0.2, 0.4]:
                    m_val = 0.0
                    err = 0.0
                    for i in range(len(log_strikes)):
                        w = svi_total_variance(log_strikes[i], a, b, rho, m_val, sigma_val)
                        if w <= 0:
                            err += 100.0
                            continue
                        model_vol = math.sqrt(max(w, 1e-12) / expiry)
                        err += (model_vol - market_vols[i]) ** 2
                    if err < best_err:
                        best_err = err
                        best_params = (a, b, rho, m_val, sigma_val)

    # Local refinement around best
    a0, b0, rho0, m0, sig0 = best_params
    for da in [-0.01, 0.0, 0.01]:
        for db in [-0.02, 0.0, 0.02]:
            for drho in [-0.1, 0.0, 0.1]:
                a_t = max(1e-6, a0 + da)
                b_t = max(1e-4, b0 + db)
                rho_t = max(-0.999, min(0.999, rho0 + drho))
                err = sum(
                    (math.sqrt(max(svi_total_variance(log_strikes[i], a_t, b_t, rho_t, m0, sig0), 1e-12) / expiry) - market_vols[i]) ** 2
                    for i in range(len(log_strikes))
                )
                if err < best_err:
                    best_err = err
                    best_params = (a_t, b_t, rho_t, m0, sig0)

    return SVIParams(a=best_params[0], b=best_params[1], rho=best_params[2],
                      m=best_params[3], sigma=best_params[4], expiry=expiry)


# ── Interpolation ────────────────────────────────────────────────────────


def interpolate_surface(
    surface: pd.DataFrame,
    strike: float,
    expiry_days: float,
) -> float:
    """Bilinear interpolation on strike x expiry IV grid."""
    if surface.empty:
        return 0.0

    strikes = surface.index.values.astype(float)
    expiries = np.array(surface.columns, dtype=float)

    strike = np.clip(strike, strikes.min(), strikes.max())
    expiry_days = np.clip(expiry_days, expiries.min(), expiries.max())

    ki = np.searchsorted(strikes, strike, side="right")
    ki = np.clip(ki, 1, len(strikes) - 1)
    k0, k1 = strikes[ki - 1], strikes[ki]
    kw = (strike - k0) / (k1 - k0) if k1 != k0 else 0.0

    ei = np.searchsorted(expiries, expiry_days, side="right")
    ei = np.clip(ei, 1, len(expiries) - 1)
    e0, e1 = expiries[ei - 1], expiries[ei]
    ew = (expiry_days - e0) / (e1 - e0) if e1 != e0 else 0.0

    v00 = surface.iloc[ki - 1, ei - 1]
    v01 = surface.iloc[ki - 1, ei]
    v10 = surface.iloc[ki, ei - 1]
    v11 = surface.iloc[ki, ei]

    vals = [v00, v01, v10, v11]
    if any(np.isnan(v) for v in vals):
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    return float((v00 * (1 - kw) + v10 * kw) * (1 - ew) + (v01 * (1 - kw) + v11 * kw) * ew)


# ── Arbitrage checks ────────────────────────────────────────────────────


def check_butterfly(surface: pd.DataFrame) -> List[ArbitrageCheck]:
    """Check convexity in strike dimension (butterfly constraint)."""
    violations: List[ArbitrageCheck] = []
    for exp in surface.columns:
        col = surface[exp].dropna()
        if len(col) < 3:
            continue
        vals = col.values
        for i in range(1, len(vals) - 1):
            bf = vals[i - 1] + vals[i + 1] - 2 * vals[i]
            if bf < -0.005:
                violations.append(ArbitrageCheck(
                    violation_type="butterfly", strike=float(col.index[i]),
                    expiry_1=float(exp),
                    detail=f"butterfly={bf:.6f}",
                ))
    return violations


def check_calendar(surface: pd.DataFrame) -> List[ArbitrageCheck]:
    """Check total variance is non-decreasing in expiry."""
    violations: List[ArbitrageCheck] = []
    expiries = sorted(surface.columns)
    for k in surface.index:
        row = surface.loc[k].dropna()
        if len(row) < 2:
            continue
        for i in range(len(row) - 1):
            t1, t2 = float(row.index[i]), float(row.index[i + 1])
            tv1 = row.iloc[i] ** 2 * t1 / 365
            tv2 = row.iloc[i + 1] ** 2 * t2 / 365
            if tv2 < tv1 - 1e-6:
                violations.append(ArbitrageCheck(
                    violation_type="calendar", strike=float(k),
                    expiry_1=t1, expiry_2=t2,
                    detail=f"tv({t1:.0f})={tv1:.6f} > tv({t2:.0f})={tv2:.6f}",
                ))
    return violations


# ── Skew / kurtosis ─────────────────────────────────────────────────────


def extract_skew_kurtosis(
    strikes: np.ndarray,
    vols: np.ndarray,
    forward: float,
    expiry_days: int,
) -> SkewKurtosis:
    """Extract skew and kurtosis from smile shape."""
    if len(strikes) < 3:
        return SkewKurtosis(expiry_days=expiry_days, atm_vol=0.0, skew_25d=0.0,
                             butterfly_25d=0.0, implied_skewness=0.0,
                             implied_kurtosis=0.0, skew_slope=0.0)

    atm_idx = int(np.argmin(np.abs(strikes - forward)))
    atm_vol = float(vols[atm_idx])

    # 25-delta approx: ~10% OTM
    put_vol = float(np.interp(forward * 0.90, strikes, vols))
    call_vol = float(np.interp(forward * 1.10, strikes, vols))

    skew = put_vol - call_vol
    butterfly = (put_vol + call_vol) / 2 - atm_vol

    # Slope at ATM
    log_m = np.log(strikes / forward)
    if atm_idx > 0 and atm_idx < len(strikes) - 1:
        dk = log_m[atm_idx + 1] - log_m[atm_idx - 1]
        slope = float((vols[atm_idx + 1] - vols[atm_idx - 1]) / dk) if abs(dk) > 1e-12 else 0.0
    else:
        slope = 0.0

    # Implied skewness ~ 3 * slope / atm_vol (Backus-Foresi-Wu approximation)
    impl_skew = 3.0 * slope / atm_vol if atm_vol > 1e-6 else 0.0

    # Implied kurtosis ~ 12 * butterfly / atm_vol
    impl_kurt = 12.0 * butterfly / atm_vol if atm_vol > 1e-6 else 3.0

    return SkewKurtosis(
        expiry_days=expiry_days, atm_vol=atm_vol,
        skew_25d=skew, butterfly_25d=butterfly,
        implied_skewness=impl_skew, implied_kurtosis=impl_kurt + 3.0,  # excess → total
        skew_slope=slope,
    )


# ── Surface dynamics ─────────────────────────────────────────────────────


def detect_surface_dynamics(
    surface_t0: pd.DataFrame,
    surface_t1: pd.DataFrame,
    forward_t0: float,
    forward_t1: float,
) -> SurfaceDynamics:
    """Detect sticky-strike vs sticky-delta regime.

    Sticky-strike: IV at fixed K stays constant when spot moves.
    Sticky-delta: IV at fixed moneyness (K/F) stays constant.
    """
    common_exp = set(surface_t0.columns) & set(surface_t1.columns)
    if not common_exp:
        return SurfaceDynamics("mixed", 0.5, 0.5, 0.0)

    exp = sorted(common_exp)[0]
    s0 = surface_t0[exp].dropna()
    s1 = surface_t1[exp].dropna()
    common_k = s0.index.intersection(s1.index)

    if len(common_k) < 5:
        return SurfaceDynamics("mixed", 0.5, 0.5, 0.0)

    iv0 = s0.loc[common_k].values
    iv1 = s1.loc[common_k].values
    iv_change = iv1 - iv0

    # Sticky-strike: IV change uncorrelated with strike
    strikes = common_k.values.astype(float)
    if np.std(iv_change) < 1e-12:
        # No IV change → sticky-strike by definition
        return SurfaceDynamics("sticky_strike", 1.0, 0.0, 0.8)

    corr_mat = np.corrcoef(strikes, iv_change)
    corr_strike = abs(float(corr_mat[0, 1])) if not np.isnan(corr_mat[0, 1]) else 0.0

    # Sticky-delta: IV change correlated with moneyness shift
    moneyness_shift = np.log(strikes / forward_t1) - np.log(strikes / forward_t0)
    if np.std(moneyness_shift) < 1e-12:
        corr_delta = 0.0
    else:
        cd = np.corrcoef(moneyness_shift, iv_change)
        corr_delta = abs(float(cd[0, 1])) if not np.isnan(cd[0, 1]) else 0.0

    if corr_strike < 0.3 and corr_delta < 0.3:
        regime = "sticky_strike"
        conf = 1.0 - corr_strike
    elif corr_delta > corr_strike:
        regime = "sticky_delta"
        conf = corr_delta
    else:
        regime = "mixed"
        conf = 0.5

    return SurfaceDynamics(
        regime=regime, sticky_strike_score=1.0 - corr_strike,
        sticky_delta_score=corr_delta, confidence=conf,
    )


# ── IV forecast ──────────────────────────────────────────────────────────


def forecast_iv(
    term_structure: List[TermStructurePoint],
    skew_list: List[SkewKurtosis],
) -> VolForecast:
    """Forecast near-term IV from surface shape."""
    if not term_structure:
        return VolForecast(0.0, 0.0, 0.0, 0.0, 0.0)

    current_atm = term_structure[0].atm_vol
    ts_slope = 0.0
    if len(term_structure) >= 2:
        dt = term_structure[-1].expiry_days - term_structure[0].expiry_days
        dv = term_structure[-1].atm_vol - term_structure[0].atm_vol
        ts_slope = dv / dt if dt > 0 else 0.0

    skew_signal = 0.0
    if skew_list:
        skew_signal = skew_list[0].implied_skewness

    # Simple forecast: mean-revert term structure + skew risk premium
    # If backwardation (short-term > long-term), IV likely to decline
    forecast = current_atm + ts_slope * 30  # 30-day horizon
    forecast = max(forecast, current_atm * 0.5)  # floor

    confidence = min(1.0, len(term_structure) / 5.0)

    return VolForecast(
        current_atm_vol=current_atm,
        forecast_atm_vol=forecast,
        term_structure_slope=ts_slope,
        skew_signal=skew_signal,
        confidence=confidence,
    )


# ── Core modeler ─────────────────────────────────────────────────────────


class VolSurfaceModeler:
    """Implied volatility surface modeler with SVI calibration."""

    def __init__(self):
        pass

    @staticmethod
    def build_surface(
        chain: pd.DataFrame,
        strike_col: str = "strike",
        expiry_col: str = "expiry_days",
        iv_col: str = "iv",
    ) -> pd.DataFrame:
        """Build strike x expiry IV grid from option chain."""
        if chain.empty:
            return pd.DataFrame()
        required = {strike_col, expiry_col, iv_col}
        if not required.issubset(chain.columns):
            return pd.DataFrame()
        surface = chain.pivot_table(
            index=strike_col, columns=expiry_col, values=iv_col, aggfunc="mean",
        )
        return surface.sort_index()

    def analyze(
        self,
        chain: pd.DataFrame,
        forward: float,
        surface_t0: Optional[pd.DataFrame] = None,
        forward_t0: Optional[float] = None,
        strike_col: str = "strike",
        expiry_col: str = "expiry_days",
        iv_col: str = "iv",
    ) -> SurfaceResult:
        """Run full surface analysis."""
        surface = self.build_surface(chain, strike_col, expiry_col, iv_col)
        if surface.empty:
            return SurfaceResult(
                surface=surface, svi_params_by_expiry={},
                skew_kurtosis=[], term_structure=[], is_contango=True,
                arbitrage_checks=[], dynamics=None, forecast=None,
                n_strikes=0, n_expiries=0,
            )

        expiries = sorted(surface.columns)
        strikes_arr = surface.index.values.astype(float)

        # SVI calibration per expiry
        svi_by_exp: Dict[int, SVIParams] = {}
        for exp in expiries:
            col = surface[exp].dropna()
            if len(col) < 3:
                continue
            ks = col.index.values.astype(float)
            vs = col.values.astype(float)
            log_k = np.log(ks / forward)
            svi = calibrate_svi(log_k, vs, float(exp) / 365.0)
            svi_by_exp[int(exp)] = svi

        # Skew / kurtosis per expiry
        sk_list: List[SkewKurtosis] = []
        for exp in expiries:
            col = surface[exp].dropna()
            if len(col) >= 3:
                sk = extract_skew_kurtosis(
                    col.index.values.astype(float), col.values.astype(float),
                    forward, int(exp),
                )
                sk_list.append(sk)

        # Term structure
        ts: List[TermStructurePoint] = []
        for exp in expiries:
            col = surface[exp].dropna()
            if col.empty:
                continue
            atm_idx = int(np.argmin(np.abs(col.index.values.astype(float) - forward)))
            atm_vol = float(col.iloc[atm_idx])
            ts.append(TermStructurePoint(
                expiry_days=int(exp), atm_vol=atm_vol,
                total_variance=atm_vol ** 2 * int(exp) / 365,
            ))
        is_contango = ts[-1].atm_vol >= ts[0].atm_vol if len(ts) >= 2 else True

        # Arbitrage checks
        violations = check_butterfly(surface) + check_calendar(surface)

        # Dynamics
        dynamics = None
        if surface_t0 is not None and forward_t0 is not None:
            dynamics = detect_surface_dynamics(surface_t0, surface, forward_t0, forward)

        # Forecast
        vol_forecast = forecast_iv(ts, sk_list)

        return SurfaceResult(
            surface=surface, svi_params_by_expiry=svi_by_exp,
            skew_kurtosis=sk_list, term_structure=ts,
            is_contango=is_contango, arbitrage_checks=violations,
            dynamics=dynamics, forecast=vol_forecast,
            n_strikes=len(strikes_arr), n_expiries=len(expiries),
        )

    @staticmethod
    def generate_report(
        result: SurfaceResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _surface_3d_svg(surface: pd.DataFrame) -> str:
    """Pseudo-3D surface plot as layered smile lines per expiry."""
    if surface.empty:
        return ""

    w, h = 700, 350
    pad = 60
    expiries = sorted(surface.columns)
    strikes = surface.index.values.astype(float)
    n_exp = len(expiries)

    if n_exp == 0 or len(strikes) < 2:
        return ""

    all_vols = surface.values.flatten()
    all_vols = all_vols[~np.isnan(all_vols)]
    if len(all_vols) == 0:
        return ""
    y_min = float(all_vols.min()) * 0.9
    y_max = float(all_vols.max()) * 1.1
    if y_max <= y_min:
        y_max = y_min + 0.01
    k_min, k_max = float(strikes.min()), float(strikes.max())

    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
              "#f0883e", "#8b949e", "#da3633"]

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">Volatility Surface (Smile per Expiry)</text>')

    pw = w - 2 * pad
    ph = h - 80

    def tx(k): return pad + (k - k_min) / (k_max - k_min) * pw if k_max > k_min else pad + pw / 2
    def ty(v): return 40 + (1 - (v - y_min) / (y_max - y_min)) * ph

    for ei, exp in enumerate(expiries):
        col = surface[exp].dropna()
        if len(col) < 2:
            continue
        color = colors[ei % len(colors)]
        ks = col.index.values.astype(float)
        vs = col.values.astype(float)
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(ks[i]):.1f},{ty(vs[i]):.1f}" for i in range(len(ks)))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8"/>')
        # Legend
        lx = pad + ei * 80
        parts.append(f'<rect x="{lx}" y="{h - 18}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{h - 9}" font-size="8" fill="#8b949e">{exp}d</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _term_structure_svg(ts: List[TermStructurePoint], is_contango: bool) -> str:
    if len(ts) < 2:
        return ""
    w, h = 600, 220
    pad = 55
    xs = [p.expiry_days for p in ts]
    ys = [p.atm_vol for p in ts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys) * 0.95, max(ys) * 1.05
    if x_max <= x_min: x_max = x_min + 1
    if y_max <= y_min: y_max = y_min + 0.01
    pw = w - 2 * pad
    ph = h - 65

    def tx(v): return pad + (v - x_min) / (x_max - x_min) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    label = "Contango" if is_contango else "Backwardation"
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">ATM Term Structure ({label})</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(xs[i]):.1f},{ty(ys[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<path d="{d}" fill="none" stroke="#58a6ff" stroke-width="2"/>')
    for i in range(len(xs)):
        parts.append(f'<circle cx="{tx(xs[i]):.1f}" cy="{ty(ys[i]):.1f}" r="4" fill="#58a6ff"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _skew_chart_svg(sk_list: List[SkewKurtosis]) -> str:
    if len(sk_list) < 2:
        return ""
    w, h = 600, 220
    pad = 55
    xs = [s.expiry_days for s in sk_list]
    ys = [s.skew_25d for s in sk_list]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys) * 1.1, max(ys) * 1.1
    if x_max <= x_min: x_max = x_min + 1
    if y_max <= y_min: y_max = y_min + 0.01
    pw = w - 2 * pad
    ph = h - 65

    def tx(v): return pad + (v - x_min) / (x_max - x_min) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">25-Delta Skew by Expiry</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(xs[i]):.1f},{ty(ys[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<path d="{d}" fill="none" stroke="#f85149" stroke-width="2"/>')
    for i in range(len(xs)):
        parts.append(f'<circle cx="{tx(xs[i]):.1f}" cy="{ty(ys[i]):.1f}" r="4" fill="#f85149"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svi_table(svi_by_exp: Dict[int, SVIParams]) -> str:
    if not svi_by_exp:
        return ""
    rows = ""
    for exp in sorted(svi_by_exp.keys()):
        p = svi_by_exp[exp]
        rows += f"<tr><td>{exp}</td><td>{_f(p.a)}</td><td>{_f(p.b)}</td><td>{_f(p.rho)}</td><td>{_f(p.m)}</td><td>{_f(p.sigma)}</td></tr>"
    return f"""<table class="data-table"><tr><th>Expiry</th><th>a</th><th>b</th><th>&rho;</th><th>m</th><th>&sigma;</th></tr>{rows}</table>"""


def _sk_table(sk_list: List[SkewKurtosis]) -> str:
    if not sk_list:
        return ""
    rows = ""
    for s in sk_list:
        rows += f"<tr><td>{s.expiry_days}</td><td>{_f(s.atm_vol)}</td><td>{_f(s.skew_25d)}</td><td>{_f(s.butterfly_25d)}</td><td>{_f(s.implied_skewness, 2)}</td><td>{_f(s.implied_kurtosis, 2)}</td></tr>"
    return f"""<table class="data-table"><tr><th>Expiry</th><th>ATM Vol</th><th>Skew</th><th>Butterfly</th><th>Skewness</th><th>Kurtosis</th></tr>{rows}</table>"""


def _build_html(result: SurfaceResult) -> str:
    fc = result.forecast
    fc_html = ""
    if fc and fc.current_atm_vol > 0:
        fc_html = f"""<div class="card"><h3>IV Forecast (30d)</h3>
        <div class="metrics-grid">
        <div><span class="label">Current ATM</span><span class="value">{_f(fc.current_atm_vol)}</span></div>
        <div><span class="label">Forecast ATM</span><span class="value">{_f(fc.forecast_atm_vol)}</span></div>
        <div><span class="label">TS Slope</span><span class="value">{_f(fc.term_structure_slope, 6)}</span></div>
        <div><span class="label">Skew Signal</span><span class="value">{_f(fc.skew_signal, 2)}</span></div>
        <div><span class="label">Confidence</span><span class="value">{_f(fc.confidence, 2)}</span></div>
        </div></div>"""

    dyn_html = ""
    if result.dynamics:
        d = result.dynamics
        dyn_html = f"""<div class="card"><h3>Surface Dynamics</h3>
        <p>Regime: <strong>{d.regime}</strong> (confidence: {_f(d.confidence, 2)})</p>
        <p>Sticky-strike score: {_f(d.sticky_strike_score, 3)} | Sticky-delta score: {_f(d.sticky_delta_score, 3)}</p></div>"""

    n_violations = len(result.arbitrage_checks)
    viol_html = ""
    if n_violations > 0:
        rows = "".join(f"<tr><td>{v.violation_type}</td><td>{v.strike:.2f}</td><td>{v.expiry_1:.0f}</td><td>{v.detail}</td></tr>" for v in result.arbitrage_checks[:20])
        viol_html = f"""<h2>Arbitrage Violations ({n_violations})</h2>
        <table class="data-table"><tr><th>Type</th><th>Strike</th><th>Expiry</th><th>Detail</th></tr>{rows}</table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Vol Surface Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Implied Volatility Surface</h1>
<p class="meta">{result.n_strikes} strikes &times; {result.n_expiries} expiries &middot;
   {'Contango' if result.is_contango else 'Backwardation'} &middot;
   {n_violations} arb violations &middot;
   {len(result.svi_params_by_expiry)} SVI calibrations</p>

{_surface_3d_svg(result.surface)}

<h2>SVI Calibration</h2>
{_svi_table(result.svi_params_by_expiry)}

<div class="two-col">
  <div>{_term_structure_svg(result.term_structure, result.is_contango)}</div>
  <div>{_skew_chart_svg(result.skew_kurtosis)}</div>
</div>

<h2>Skew &amp; Kurtosis</h2>
{_sk_table(result.skew_kurtosis)}

<div class="two-col">
  {fc_html}
  {dyn_html}
</div>

{viol_html}

</body>
</html>"""
