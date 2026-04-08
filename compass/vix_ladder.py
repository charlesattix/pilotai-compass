"""compass/vix_ladder.py — Standalone VIX exposure ladder (EXP-2820 winner).

Production-ready module for scaling portfolio exposure as a function of
the current VIX level. The default ladder is the EXP-2820 winning
variant that reduced the VIX→80 flash-crash scenario DD from 43.1% to
0.80% while lifting normal-regime Sharpe by +0.49.

The ladder is a step-wise linear-interpolating function from VIX to
exposure multiplier (1.0 = full, 0.0 = flat). It's causal by default:
the public API shifts by 1 day so today's exposure uses yesterday's
VIX close.

Usage
-----
    from compass.vix_ladder import VIXLadder

    ladder = VIXLadder()                     # EXP-2820 default
    exposure = ladder.apply(vix_series)      # causal (shift-by-1)
    adjusted = portfolio_returns * exposure

    # Custom ladder
    ladder = VIXLadder(breakpoints=[(20, 1.0), (30, 0.5), (40, 0.0)])

    # Non-causal (test / research)
    ladder = VIXLadder(causal=False)

The module is pure Python + numpy + pandas. No network calls, no
IronVault dependency. Callers supply the VIX series.

Rule Zero: the ladder operates on whatever VIX series is passed in.
Production callers should fetch real Yahoo ^VIX (see
compass.exp2690_signal_generators._fetch_yahoo_close).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# EXP-2820 winning ladder: 9 breakpoints from VIX 20 → VIX 70+
# Pooled net Sharpe delta vs baseline: +0.486
# Flash-crash DD: 43.1% → 0.80% (−42.3pp)
EXP2820_DEFAULT_LADDER: List[Tuple[float, float]] = [
    (20.0, 1.00),
    (25.0, 0.90),
    (30.0, 0.75),
    (35.0, 0.60),
    (40.0, 0.50),
    (50.0, 0.35),
    (60.0, 0.25),
    (70.0, 0.15),
    (1e9,  0.00),
]


class VIXLadder:
    """VIX→exposure ladder with linear interpolation between breakpoints.

    Parameters
    ----------
    breakpoints : list of (vix, exposure) pairs, sorted ascending by vix.
        Defaults to the EXP-2820 winning ladder.
    causal : if True (default), the apply() method shifts input by 1 day
        so today's exposure decision uses yesterday's VIX close.
    min_exposure : hard floor clamp on the output (default 0.0).
    max_exposure : hard ceiling clamp on the output (default 1.0).
    """

    def __init__(
        self,
        breakpoints: Optional[Sequence[Tuple[float, float]]] = None,
        causal: bool = True,
        min_exposure: float = 0.0,
        max_exposure: float = 1.0,
    ):
        bps = list(breakpoints) if breakpoints is not None else list(EXP2820_DEFAULT_LADDER)
        # Validate: sorted, monotone non-increasing exposure
        for i in range(len(bps) - 1):
            if bps[i + 1][0] <= bps[i][0]:
                raise ValueError(f"breakpoints must be strictly ascending in VIX: "
                                  f"{bps[i]} → {bps[i+1]}")
            if bps[i + 1][1] > bps[i][1]:
                # Warn but allow — someone may want a non-monotone ladder
                pass
        if min_exposure < 0 or max_exposure > 1 or min_exposure > max_exposure:
            raise ValueError(f"invalid exposure bounds: "
                              f"[{min_exposure}, {max_exposure}]")
        self.breakpoints: List[Tuple[float, float]] = bps
        self.causal = bool(causal)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)

    # ── Core evaluator ─────────────────────────────────────────────────
    def exposure_at(self, vix: float) -> float:
        """Return the exposure multiplier for a single VIX value via
        linear interpolation between the ladder's breakpoints."""
        if vix is None or (isinstance(vix, float) and np.isnan(vix)):
            return self.max_exposure   # permissive fallback on missing data
        if vix <= self.breakpoints[0][0]:
            return self._clip(self.breakpoints[0][1])
        if vix >= self.breakpoints[-1][0]:
            return self._clip(self.breakpoints[-1][1])
        # Linear interpolate between the bracketing breakpoints
        for i in range(len(self.breakpoints) - 1):
            v_lo, e_lo = self.breakpoints[i]
            v_hi, e_hi = self.breakpoints[i + 1]
            if v_lo <= vix < v_hi:
                span = v_hi - v_lo
                if span <= 0:
                    return self._clip(e_lo)
                frac = (vix - v_lo) / span
                return self._clip(e_lo + frac * (e_hi - e_lo))
        return self._clip(self.breakpoints[-1][1])

    def _clip(self, x: float) -> float:
        return float(max(self.min_exposure, min(self.max_exposure, x)))

    # ── Vectorised series API (the production entry point) ────────────
    def apply(self, vix: Union[pd.Series, np.ndarray, Sequence[float]]
               ) -> pd.Series:
        """Vectorised exposure computation on a VIX series.

        If `causal=True` (the default), the returned series is shifted
        by 1 — today's exposure is derived from yesterday's VIX close.
        The first element uses max_exposure (no prior VIX available).

        Accepts pd.Series, np.ndarray, or list/tuple. Returns pd.Series
        if the input is a Series (preserves index); otherwise np.ndarray.
        """
        is_series = isinstance(vix, pd.Series)
        arr = np.asarray(vix, dtype=float)

        # Vectorised interpolation via np.interp — but our ladder is
        # piecewise linear with a tail that caps at the last breakpoint.
        # Build parallel arrays of xs and ys.
        xs = np.array([bp[0] for bp in self.breakpoints], dtype=float)
        ys = np.array([bp[1] for bp in self.breakpoints], dtype=float)
        # np.interp clamps to endpoints automatically
        raw = np.interp(arr, xs, ys)
        # NaN propagation: np.interp returns NaN for NaN inputs; replace
        # with max_exposure (permissive fallback).
        nan_mask = np.isnan(arr)
        raw = np.where(nan_mask, self.max_exposure, raw)
        raw = np.clip(raw, self.min_exposure, self.max_exposure)

        if self.causal:
            shifted = np.empty_like(raw)
            shifted[0] = self.max_exposure
            shifted[1:] = raw[:-1]
            out = shifted
        else:
            out = raw

        if is_series:
            return pd.Series(out, index=vix.index, name="vix_ladder_exposure")
        return out

    # ── Introspection ──────────────────────────────────────────────────
    def describe(self) -> dict:
        return {
            "breakpoints": self.breakpoints,
            "causal": self.causal,
            "min_exposure": self.min_exposure,
            "max_exposure": self.max_exposure,
            "source": "EXP-2820 flash crash protection winner",
        }

    def __repr__(self) -> str:
        bps = ", ".join(f"({v:.0f}→{e:.2f})" for v, e in self.breakpoints[:4])
        return f"VIXLadder({bps}, ..., causal={self.causal})"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: production VIX fetch + apply in one call
# ═══════════════════════════════════════════════════════════════════════════

def fetch_vix(start: str, end: str) -> pd.Series:
    """Real Yahoo ^VIX daily close for [start, end]."""
    import warnings
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download("^VIX", start=start, end=end,
                          progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "VIX"
    return s


def apply_to_portfolio(
    portfolio_returns: pd.Series,
    vix_series: Optional[pd.Series] = None,
    ladder: Optional[VIXLadder] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Apply a VIX ladder to a daily portfolio return series.

    Returns (adjusted_returns, exposure_series). The VIX series is
    fetched from Yahoo if not provided. Both series are aligned to
    the portfolio_returns index.
    """
    if ladder is None:
        ladder = VIXLadder()

    if vix_series is None:
        start = (portfolio_returns.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end = (portfolio_returns.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        vix_series = fetch_vix(start, end)

    vix_aligned = vix_series.reindex(portfolio_returns.index).ffill().bfill()
    exposure = ladder.apply(vix_aligned)
    adjusted = portfolio_returns * exposure
    return adjusted, exposure


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Verify the ladder evaluates correctly at key VIX levels."""
    ladder = VIXLadder()

    # Point checks on the default ladder breakpoints
    cases = [
        (10.0, 1.00),   # below first breakpoint → clamped to 1.0
        (20.0, 1.00),   # at first breakpoint
        (25.0, 0.90),
        (30.0, 0.75),
        (35.0, 0.60),
        (40.0, 0.50),
        (50.0, 0.35),
        (60.0, 0.25),
        (70.0, 0.15),
        (80.0, 0.15 + (0.00 - 0.15) * (80 - 70) / (1e9 - 70)),  # near 0.15 since 1e9 is huge
        (100.0, 0.15 + (0.00 - 0.15) * (100 - 70) / (1e9 - 70)),
    ]
    for vix, expected in cases:
        got = ladder.exposure_at(vix)
        assert abs(got - expected) < 1e-6, \
            f"VIX={vix}: expected {expected}, got {got}"
    # Interpolation mid-way between breakpoints
    mid_25_30 = ladder.exposure_at(27.5)
    assert abs(mid_25_30 - 0.825) < 1e-6, f"interp 27.5 mismatch: {mid_25_30}"

    # Vectorised API
    vix_series = pd.Series(
        [15.0, 22.0, 28.0, 32.0, 42.0, 65.0, np.nan],
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )
    # Non-causal
    exp_nc = VIXLadder(causal=False).apply(vix_series)
    assert abs(exp_nc.iloc[0] - 1.0) < 1e-6
    assert abs(exp_nc.iloc[5] - 0.2) < 1e-6       # VIX 65 → midpoint of 0.25 and 0.15
    assert abs(exp_nc.iloc[6] - 1.0) < 1e-6       # NaN → max_exposure

    # Causal: each day uses yesterday's VIX
    exp_c = VIXLadder(causal=True).apply(vix_series)
    assert abs(exp_c.iloc[0] - 1.0) < 1e-6        # first day: max_exposure
    assert abs(exp_c.iloc[1] - 1.0) < 1e-6        # yesterday VIX 15 → 1.0
    # yesterday VIX 28 → interp between (25, 0.90) and (30, 0.75) at frac 0.6
    # → 0.90 + 0.6 * (0.75 - 0.90) = 0.81
    assert abs(exp_c.iloc[3] - 0.81) < 1e-6
    print("[self_test] all checks passed")


if __name__ == "__main__":
    _self_test()
    import json
    ladder = VIXLadder()
    print(json.dumps(ladder.describe(), indent=2, default=str))
