"""
Momentum crash protector — detects crowding and reversal risk.

Indicators:
  1. Cross-sectional momentum dispersion
  2. Return autocorrelation decay
  3. Winner-loser spread acceleration
  4. Composite crowding score
  5. Short interest proxy (loser vol / winner vol)
  6. Mean reversion trigger (3-day W-L flip)

Usage::

    from compass.momentum_crash_protector import MomentumCrashProtector
    protector = MomentumCrashProtector(returns_df)
    result = protector.analyze()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "momentum_crash_protector.html"
TRADING_DAYS = 252


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class CrashIndicators:
    """Daily crash risk indicators."""

    date: Any
    momentum_dispersion: float    # cross-sectional std of momentum
    autocorrelation: float        # rolling return autocorrelation
    wl_spread: float              # winner - loser return
    wl_acceleration: float        # d/dt of wl_spread
    short_interest_proxy: float   # loser_vol / winner_vol
    mean_reversion_trigger: bool  # winners underperforming losers 3d
    crowding_score: float         # composite 0-1
    risk_level: str               # "low", "elevated", "high", "critical"


@dataclass
class CrashEpisode:
    """Detected crash episode."""

    start_date: Any
    peak_crowding: float
    pre_crash_signal_days: int  # days of elevated signal before crash
    crash_magnitude_pct: float
    detected_early: bool


@dataclass
class ProtectionResult:
    """Result of applying crash protection."""

    unprotected_return: float
    unprotected_dd: float
    protected_return: float
    protected_dd: float
    dd_reduction_pct: float
    return_preserved_pct: float
    n_protection_days: int
    n_contrarian_days: int


@dataclass
class AnalysisResult:
    """Full crash protection analysis."""

    indicators: pd.DataFrame
    episodes: List[CrashEpisode]
    protection: ProtectionResult
    avg_crowding: float
    max_crowding: float
    n_elevated_days: int
    n_critical_days: int
    episodes_detected_early: int
    total_episodes: int
    n_observations: int


# ── Indicator computations ───────────────────────────────────────────────


def compute_momentum(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling cumulative returns (momentum) per asset."""
    return returns.rolling(window).sum()


def momentum_dispersion(momentum: pd.DataFrame) -> pd.Series:
    """Cross-sectional standard deviation of momentum scores."""
    return momentum.std(axis=1).fillna(0)


def return_autocorrelation(
    returns: pd.Series, window: int = 20,
) -> pd.Series:
    """Rolling lag-1 autocorrelation of returns."""
    result = pd.Series(0.0, index=returns.index)
    vals = returns.values
    for i in range(window + 1, len(vals)):
        x = vals[i - window:i - 1]
        y = vals[i - window + 1:i]
        if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        result.iloc[i] = float(np.corrcoef(x, y)[0, 1])
    return result


def winner_loser_spread(
    returns: pd.DataFrame, momentum: pd.DataFrame,
) -> pd.Series:
    """Daily return of winners minus losers (momentum factor)."""
    result = pd.Series(0.0, index=returns.index)
    for i in range(len(returns)):
        mom_row = momentum.iloc[i].dropna()
        if len(mom_row) < 2:
            continue
        median = mom_row.median()
        winners = mom_row.index[mom_row > median]
        losers = mom_row.index[mom_row <= median]
        if len(winners) == 0 or len(losers) == 0:
            continue
        w_ret = returns.iloc[i][winners].mean()
        l_ret = returns.iloc[i][losers].mean()
        result.iloc[i] = w_ret - l_ret
    return result


def wl_spread_acceleration(wl_spread: pd.Series, window: int = 5) -> pd.Series:
    """Rate of change of winner-loser spread."""
    return wl_spread.diff(window).fillna(0)


def short_interest_proxy(
    returns: pd.DataFrame, momentum: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Loser volume / winner volume ratio. Uses return magnitude as vol proxy."""
    result = pd.Series(1.0, index=returns.index)
    for i in range(len(returns)):
        mom_row = momentum.iloc[i].dropna()
        if len(mom_row) < 2:
            continue
        median = mom_row.median()
        winners = mom_row.index[mom_row > median]
        losers = mom_row.index[mom_row <= median]
        if len(winners) == 0 or len(losers) == 0:
            continue
        # Use absolute return as volume proxy
        w_activity = returns.iloc[i][winners].abs().mean()
        l_activity = returns.iloc[i][losers].abs().mean()
        if w_activity > 1e-12:
            result.iloc[i] = l_activity / w_activity
    return result


def mean_reversion_trigger(
    returns: pd.DataFrame, momentum: pd.DataFrame, lookback: int = 3,
) -> pd.Series:
    """True when winners underperform losers over lookback days."""
    wl = winner_loser_spread(returns, momentum)
    rolling_wl = wl.rolling(lookback).sum()
    return (rolling_wl < 0).astype(float)


# ── Composite crowding score ─────────────────────────────────────────────


def compute_crowding_score(
    dispersion_z: float,
    autocorr: float,
    wl_accel: float,
    si_proxy: float,
) -> float:
    """Composite crowding score (0-1). Higher = more crash risk.

    Components:
      - High dispersion → crowded (weight 0.3)
      - Low/decaying autocorrelation → reversal coming (weight 0.25)
      - Negative WL acceleration → spread collapsing (weight 0.25)
      - High loser activity → short squeeze risk (weight 0.2)
    """
    # Dispersion: z-score > 1.5 is crowded
    disp_score = min(max(dispersion_z / 3.0, 0), 1)

    # Autocorrelation: low = bad for momentum
    acorr_score = min(max((0.3 - autocorr) / 0.6, 0), 1)

    # WL acceleration: negative = spread collapsing
    wl_score = min(max(-wl_accel * 50, 0), 1)

    # Short interest: high = squeeze risk
    si_score = min(max((si_proxy - 1.0) / 2.0, 0), 1)

    return 0.30 * disp_score + 0.25 * acorr_score + 0.25 * wl_score + 0.20 * si_score


def classify_risk(crowding: float) -> str:
    if crowding > 0.75:
        return "critical"
    if crowding > 0.50:
        return "high"
    if crowding > 0.30:
        return "elevated"
    return "low"


# ── Episode detection ────────────────────────────────────────────────────


def detect_episodes(
    indicators: pd.DataFrame,
    returns: pd.Series,
    crash_threshold_pct: float = 3.0,
    lookback: int = 5,
) -> List[CrashEpisode]:
    """Detect crash episodes and check if indicators warned in advance."""
    episodes = []
    # Find drawdown events
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak

    in_crash = False
    crash_start = None
    for i in range(lookback, len(dd)):
        if dd.iloc[i] < -crash_threshold_pct / 100 and not in_crash:
            in_crash = True
            crash_start = i
        elif dd.iloc[i] > -crash_threshold_pct / 200 and in_crash:
            # Crash ended
            mag = float(dd.iloc[crash_start:i].min() * 100)
            # Check if crowding was elevated before crash
            pre_crowd = indicators["crowding_score"].iloc[max(0, crash_start - lookback):crash_start]
            peak_crowd = float(pre_crowd.max()) if len(pre_crowd) > 0 else 0
            early_days = int((pre_crowd > 0.4).sum())
            episodes.append(CrashEpisode(
                start_date=indicators.index[crash_start] if hasattr(indicators.index, '__getitem__') else crash_start,
                peak_crowding=peak_crowd,
                pre_crash_signal_days=early_days,
                crash_magnitude_pct=mag,
                detected_early=early_days >= 1,
            ))
            in_crash = False

    return episodes


# ── Protection simulation ────────────────────────────────────────────────


def simulate_protection(
    returns: pd.Series,
    crowding_scores: pd.Series,
    high_threshold: float = 0.50,
    critical_threshold: float = 0.75,
) -> ProtectionResult:
    """Simulate portfolio with crash protection.

    - Low crowding: full momentum exposure
    - Elevated: reduce to 50%
    - High: reduce to 20%
    - Critical: flip to contrarian (-30% momentum, +30% market)
    """
    n = len(returns)
    protected_returns = np.zeros(n)
    n_protect = 0
    n_contrarian = 0

    for i in range(n):
        crowd = float(crowding_scores.iloc[i]) if i < len(crowding_scores) else 0
        r = float(returns.iloc[i])

        if crowd > critical_threshold:
            protected_returns[i] = -r * 0.3  # contrarian
            n_contrarian += 1
        elif crowd > high_threshold:
            protected_returns[i] = r * 0.2
            n_protect += 1
        elif crowd > 0.30:
            protected_returns[i] = r * 0.5
            n_protect += 1
        else:
            protected_returns[i] = r

    # Metrics
    unp_eq = (1 + returns).cumprod().values
    pro_eq = np.cumprod(1 + protected_returns)

    def _dd(eq):
        pk = np.maximum.accumulate(eq)
        return float(abs(((eq - pk) / pk).min()) * 100)

    def _ret(eq):
        return float((eq[-1] / eq[0] - 1) * 100) if eq[0] > 0 else 0

    unp_ret = _ret(unp_eq)
    unp_dd = _dd(unp_eq)
    pro_ret = _ret(pro_eq)
    pro_dd = _dd(pro_eq)

    dd_red = (unp_dd - pro_dd) / unp_dd * 100 if unp_dd > 0.01 else 0
    ret_pres = pro_ret / unp_ret * 100 if abs(unp_ret) > 0.01 else 100

    return ProtectionResult(
        unprotected_return=unp_ret, unprotected_dd=unp_dd,
        protected_return=pro_ret, protected_dd=pro_dd,
        dd_reduction_pct=dd_red, return_preserved_pct=ret_pres,
        n_protection_days=n_protect, n_contrarian_days=n_contrarian,
    )


# ── Core engine ──────────────────────────────────────────────────────────


class MomentumCrashProtector:
    """Momentum crash detection and protection engine."""

    def __init__(
        self,
        returns: pd.DataFrame,
        momentum_window: int = 20,
        autocorr_window: int = 20,
    ):
        if returns.empty:
            raise ValueError("returns must not be empty")
        self.returns = returns
        self.momentum_window = momentum_window
        self.autocorr_window = autocorr_window

    def analyze(self) -> AnalysisResult:
        mom = compute_momentum(self.returns, self.momentum_window)
        avg_returns = self.returns.mean(axis=1)

        # Indicators
        disp = momentum_dispersion(mom)
        disp_mean = disp.rolling(60).mean().fillna(disp.mean())
        disp_std = disp.rolling(60).std().fillna(1)
        disp_z = ((disp - disp_mean) / disp_std.replace(0, 1)).fillna(0)

        acorr = return_autocorrelation(avg_returns, self.autocorr_window)
        wl = winner_loser_spread(self.returns, mom)
        wl_acc = wl_spread_acceleration(wl)
        si = short_interest_proxy(self.returns, mom)
        mr_trigger = mean_reversion_trigger(self.returns, mom)

        # Crowding scores
        crowding = pd.Series(0.0, index=self.returns.index)
        for i in range(len(self.returns)):
            crowding.iloc[i] = compute_crowding_score(
                float(disp_z.iloc[i]),
                float(acorr.iloc[i]),
                float(wl_acc.iloc[i]),
                float(si.iloc[i]),
            )

        risk_levels = crowding.apply(classify_risk)

        indicators = pd.DataFrame({
            "dispersion": disp,
            "dispersion_z": disp_z,
            "autocorrelation": acorr,
            "wl_spread": wl,
            "wl_acceleration": wl_acc,
            "short_interest_proxy": si,
            "mean_reversion_trigger": mr_trigger,
            "crowding_score": crowding,
            "risk_level": risk_levels,
        }, index=self.returns.index)

        # Episodes
        episodes = detect_episodes(indicators, avg_returns)

        # Protection
        protection = simulate_protection(avg_returns, crowding)

        n_elevated = int((crowding > 0.30).sum())
        n_critical = int((crowding > 0.75).sum())
        early = sum(1 for e in episodes if e.detected_early)

        return AnalysisResult(
            indicators=indicators, episodes=episodes,
            protection=protection,
            avg_crowding=float(crowding.mean()),
            max_crowding=float(crowding.max()),
            n_elevated_days=n_elevated,
            n_critical_days=n_critical,
            episodes_detected_early=early,
            total_episodes=len(episodes),
            n_observations=len(self.returns),
        )

    @staticmethod
    def generate_report(result: AnalysisResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"


def _build_html(r: AnalysisResult) -> str:
    p = r.protection
    def _ep_row(e):
        c = "#3fb950" if e.detected_early else "#f85149"
        det = "YES" if e.detected_early else "NO"
        return (f"<tr><td>{e.start_date}</td><td>{_fp(e.crash_magnitude_pct)}</td>"
                f"<td>{_fr(e.peak_crowding)}</td><td>{e.pre_crash_signal_days}d</td>"
                f"<td style='color:{c}'>{det}</td></tr>")
    ep_rows = "".join(_ep_row(e) for e in r.episodes)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Momentum Crash Protection</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}</style></head><body>
<h1>Momentum Crash Protection</h1>
<div class="cards">
<div class="c"><div class="l">DD Reduction</div><div class="v">{_fp(p.dd_reduction_pct)}</div></div>
<div class="c"><div class="l">Return Preserved</div><div class="v">{_fp(p.return_preserved_pct)}</div></div>
<div class="c"><div class="l">Unprotected DD</div><div class="v">{_fp(p.unprotected_dd)}</div></div>
<div class="c"><div class="l">Protected DD</div><div class="v">{_fp(p.protected_dd)}</div></div>
<div class="c"><div class="l">Episodes Detected</div><div class="v">{r.episodes_detected_early}/{r.total_episodes}</div></div>
<div class="c"><div class="l">Elevated Days</div><div class="v">{r.n_elevated_days}</div></div>
<div class="c"><div class="l">Avg Crowding</div><div class="v">{_fr(r.avg_crowding)}</div></div>
<div class="c"><div class="l">Protection Days</div><div class="v">{p.n_protection_days}</div></div>
</div>
<h2>Crash Episodes</h2>
<table><tr><th>Date</th><th>Magnitude</th><th>Peak Crowding</th><th>Early Signal</th><th>Detected?</th></tr>{ep_rows}</table>
</body></html>"""
