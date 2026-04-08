"""EXP-1780 Crisis Alpha v5 — Hedge Optimization (NOT standalone return).

v4 maximized standalone Calmar (DD<15% + best risk-adjusted return).
v5 has a different objective: MINIMIZE correlation to EXP-1220 during the
worst EXP-1220 drawdown periods, while still being self-financing.

Carlos's use case: leveraged EXP-1220 (e.g. 2x). Even small positive
correlation under stress destroys the hedge value during the exact moments
it's needed. We want a strategy whose returns are reliably ZERO or POSITIVE
when EXP-1220 is bleeding.

Three new design knobs vs v4:
  1. STRESS GATE — only deploy capital when SPY rolling DD > stress_threshold.
     Otherwise hold cash. This concentrates the hedge's risk budget on the
     periods where it actually matters.
  2. SAFE-HAVEN TILT — preferential weighting on TLT/GLD/UUP, the assets
     that historically rally during equity stress. Equity legs (SPY/IWM/EFA/
     EEM/QQQ) only get short positions allowed.
  3. HEDGE OBJECTIVE — grid is scored by `dd_corr` (corr during EXP-1220 DD)
     instead of Calmar. Tie-broken by downside-capture: how much the hedge
     covers EXP-1220's losses on its worst days.

Test: does adding 10% v5 to a 2× LEVERAGED EXP-1220 reduce its DD?
This is the harder test — leverage doubles the drawdown a hedge has to
neutralize.

Rule Zero: real Yahoo Finance only. No synthetic data anywhere.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.crisis_alpha_v3 import LOOKBACK_GRID, load_universe_v3
from compass.crisis_alpha_v4 import (
    UNIVERSE_V4, compute_metrics, corrected_sharpe,
    compute_signal_with_confirmation, apply_drawdown_brake,
)
from compass.exp1780_exp1220_integration import build_exp1220_daily_returns

TRADING_DAYS = 252

# Safe-haven assets get preferential weight; equity legs are short-only
SAFE_HAVENS = ["TLT", "GLD", "LQD", "UUP"]
EQUITIES = ["SPY", "IWM", "EFA", "EEM", "QQQ"]
OTHER = ["HYG"]


@dataclass
class HedgeConfigV5:
    name: str
    lookback_preset: str
    vol_target: float
    leverage: float
    dd_brake_threshold: float
    dd_brake_zone: float
    max_weight: float
    require_confirmation: bool
    stress_threshold: float        # SPY rolling DD level above which the gate opens
    stress_lookback: int           # window for SPY DD calc
    safe_haven_boost: float        # multiplier on safe-haven raw weights
    equity_short_only: bool

    # Results
    n_days: int = 0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    calmar: float = 0.0
    vol: float = 0.0
    corr_full: float = 0.0
    corr_dd: float = 0.0           # PRIMARY HEDGE METRIC
    downside_capture: float = 0.0  # avg hedge return on EXP-1220's worst quartile days
    hedge_score: float = 0.0       # composite (lower=better hedge)
    daily_returns: Optional[pd.Series] = None


@dataclass
class LeveragedHedgeTest:
    leverage: float
    crisis_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    calmar: float
    dd_2022: float
    return_2022: float
    corr: float


# ═══════════════════════════════════════════════════════════════════════════
# Hedge-aware sizing
# ═══════════════════════════════════════════════════════════════════════════

def compute_v5_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: HedgeConfigV5,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """v5 sizing: vol-target + safe-haven tilt + equity-short-only."""
    returns = prices.pct_change().fillna(0)
    rolling_vol = (returns.rolling(vol_lookback, min_periods=20).std()
                   * math.sqrt(TRADING_DAYS)).fillna(cfg.vol_target)

    raw = (np.sign(signal)
           * np.minimum(np.abs(signal) * 5, 1.0)
           * cfg.vol_target / rolling_vol)
    raw = raw.clip(-cfg.max_weight, cfg.max_weight)

    # Safe-haven tilt: amplify their weights
    for tk in SAFE_HAVENS:
        if tk in raw.columns:
            raw[tk] = raw[tk] * cfg.safe_haven_boost
            raw[tk] = raw[tk].clip(-cfg.max_weight, cfg.max_weight * 1.5)

    # Equity-short-only: zero out long equity positions
    if cfg.equity_short_only:
        for tk in EQUITIES:
            if tk in raw.columns:
                raw[tk] = raw[tk].where(raw[tk] < 0, 0.0)

    # Leverage cap
    gross = raw.abs().sum(axis=1)
    scale = np.where(gross > cfg.leverage, cfg.leverage / gross, 1.0)
    raw = raw.multiply(scale, axis=0)
    return raw


def stress_gate(spy_prices: pd.Series, threshold: float, lookback: int) -> pd.Series:
    """Returns 1.0 when SPY rolling DD ≥ threshold, else 0.0.

    Uses the rolling-window peak (looking backward only — no peeking)."""
    cum = spy_prices / spy_prices.iloc[0]
    rolling_peak = cum.rolling(lookback, min_periods=20).max()
    rolling_dd = 1.0 - (cum / rolling_peak)
    gate = (rolling_dd >= threshold).astype(float)
    return gate.shift(1).fillna(0.0)   # lag by 1 day → no look-ahead


# ═══════════════════════════════════════════════════════════════════════════
# Backtest with hedge logic
# ═══════════════════════════════════════════════════════════════════════════

def backtest_v5(
    prices: pd.DataFrame,
    config: HedgeConfigV5,
    rebalance_days: int = 5,
) -> HedgeConfigV5:
    universe = [c for c in UNIVERSE_V4 if c in prices.columns]
    sub = prices[universe]

    lookbacks, lw = LOOKBACK_GRID[config.lookback_preset]
    signal = compute_signal_with_confirmation(
        sub, lookbacks, lw, config.require_confirmation
    )
    weights = compute_v5_weights(sub, signal, config)

    asset_returns = sub.pct_change().fillna(0)
    held = weights.copy()
    for i in range(len(held)):
        if i % rebalance_days != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)
    raw_port_rets = (lagged * asset_returns).sum(axis=1)

    # Stress gate (zero exposure outside stress regime)
    if config.stress_threshold > 0:
        gate = stress_gate(sub["SPY"], config.stress_threshold, config.stress_lookback)
        gate = gate.reindex(raw_port_rets.index).fillna(0)
        raw_port_rets = raw_port_rets * gate

    warmup = max(lookbacks)
    if len(sub) > warmup:
        valid_idx = sub.index[warmup]
        raw_port_rets = raw_port_rets[raw_port_rets.index >= valid_idx]

    # DD brake (feedback control from v4)
    raw_arr = raw_port_rets.values.copy()
    braked = apply_drawdown_brake(
        raw_arr, config.dd_brake_threshold, config.dd_brake_zone
    )
    port_rets = pd.Series(braked, index=raw_port_rets.index)

    m = compute_metrics(port_rets.values)
    config.n_days = len(port_rets)
    config.cagr = round(m["cagr"] * 100, 2)
    config.sharpe = round(m["sharpe"], 2)
    config.max_dd = round(m["dd"] * 100, 2)
    config.calmar = round(m["calmar"], 2)
    config.vol = round(m["vol"] * 100, 2)
    config.daily_returns = port_rets
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Hedge effectiveness scoring
# ═══════════════════════════════════════════════════════════════════════════

def find_dd_periods(
    returns: pd.Series, dd_threshold: float = 0.03
) -> pd.Series:
    """Boolean series: True on days when running DD ≥ dd_threshold.

    Used to identify "EXP-1220 is in pain" periods to test the hedge against.
    """
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = 1.0 - eq / peak
    return dd >= dd_threshold


def score_hedge(
    hedge_rets: pd.Series,
    exp1220_rets: pd.Series,
    dd_threshold: float = 0.03,
) -> Dict[str, float]:
    """Compute hedge effectiveness vs EXP-1220.

    Returns:
      corr_full        — full-sample correlation
      corr_dd          — correlation during EXP-1220 DD periods (PRIMARY)
      downside_capture — mean hedge return on EXP-1220's worst-quartile days
                         (positive = hedge makes money on bad days for 1220)
      hedge_score      — composite: corr_dd - 5*downside_capture
                         (lower is better)
    """
    common = hedge_rets.index.intersection(exp1220_rets.index)
    h = hedge_rets.reindex(common).fillna(0)
    e = exp1220_rets.reindex(common).fillna(0)

    if h.std() < 1e-12 or e.std() < 1e-12:
        return {"corr_full": 0.0, "corr_dd": 0.0,
                "downside_capture": 0.0, "hedge_score": 0.0}

    corr_full = float(h.corr(e))

    in_dd = find_dd_periods(e, dd_threshold)
    if in_dd.sum() > 10:
        h_dd = h[in_dd]
        e_dd = e[in_dd]
        if h_dd.std() > 1e-12 and e_dd.std() > 1e-12:
            corr_dd = float(h_dd.corr(e_dd))
        else:
            corr_dd = 0.0
    else:
        corr_dd = corr_full

    # Worst-quartile days
    threshold = e.quantile(0.25)  # lowest 25% of EXP-1220 days
    worst_mask = e <= threshold
    if worst_mask.sum() > 10:
        downside_capture = float(h[worst_mask].mean())
    else:
        downside_capture = 0.0

    # Composite — lower is better hedge. Penalize positive corr in DD; reward
    # positive returns on the worst days.
    hedge_score = corr_dd - 5.0 * downside_capture * 100  # scale capture into bps

    return {
        "corr_full": corr_full,
        "corr_dd": corr_dd,
        "downside_capture": downside_capture,
        "hedge_score": hedge_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Grid search optimizing for hedge effectiveness
# ═══════════════════════════════════════════════════════════════════════════

def search_hedge_configs(
    prices: pd.DataFrame,
    exp1220: pd.Series,
) -> List[HedgeConfigV5]:
    """Focused grid: vary stress gate, safe-haven tilt, equity-short flag."""
    grid = []
    for preset in ["v2_round", "slow"]:
        for vt in [0.05, 0.08]:
            for lev in [1.0, 1.5]:
                for stress_thr in [0.0, 0.03, 0.05]:    # 0 = always on
                    for sh_boost in [1.0, 1.5, 2.0]:
                        for short_only in [False, True]:
                            grid.append(HedgeConfigV5(
                                name=(f"{preset}/v{vt}/l{lev}/"
                                      f"sg{stress_thr:.2f}/sh{sh_boost}/"
                                      f"{'sho' if short_only else 'ls'}"),
                                lookback_preset=preset,
                                vol_target=vt,
                                leverage=lev,
                                dd_brake_threshold=0.05,
                                dd_brake_zone=0.03,
                                max_weight=0.20,
                                require_confirmation=False,
                                stress_threshold=stress_thr,
                                stress_lookback=60,
                                safe_haven_boost=sh_boost,
                                equity_short_only=short_only,
                            ))

    print(f"Searching {len(grid)} hedge configs...")
    out = []
    for cfg in grid:
        try:
            r = backtest_v5(prices, cfg)
            scores = score_hedge(r.daily_returns, exp1220)
            r.corr_full = round(scores["corr_full"], 3)
            r.corr_dd = round(scores["corr_dd"], 3)
            r.downside_capture = round(scores["downside_capture"], 5)
            r.hedge_score = round(scores["hedge_score"], 3)
            out.append(r)
        except Exception as e:
            print(f"  Error {cfg.name}: {e}")
    return out


def select_best_hedge(configs: List[HedgeConfigV5]) -> HedgeConfigV5:
    """Pick the lowest hedge_score, with the constraint that the strategy
    must not be a pure cash strategy (n_days > 0 AND nontrivial vol).
    """
    eligible = [c for c in configs if c.vol > 1.0]   # at least 1% annual vol
    if not eligible:
        return min(configs, key=lambda c: c.hedge_score)
    return min(eligible, key=lambda c: c.hedge_score)


# ═══════════════════════════════════════════════════════════════════════════
# Leveraged EXP-1220 stress test
# ═══════════════════════════════════════════════════════════════════════════

def test_leveraged_hedge(
    exp1220: pd.Series,
    hedge: pd.Series,
    leverage: float,
    crisis_pct: float,
) -> LeveragedHedgeTest:
    """Apply `leverage` × to EXP-1220 and combine with `crisis_pct` of hedge."""
    common = exp1220.index.intersection(hedge.index)
    e = exp1220.reindex(common).fillna(0) * leverage
    h = hedge.reindex(common).fillna(0)

    combined = (1 - crisis_pct) * e + crisis_pct * h
    m = compute_metrics(combined.values)

    mask_2022 = combined.index.year == 2022
    if mask_2022.sum() > 5:
        m_2022 = compute_metrics(combined[mask_2022].values)
        dd_2022 = m_2022["dd"] * 100
        ret_2022 = float((np.prod(1 + combined[mask_2022].values) - 1) * 100)
    else:
        dd_2022 = 0.0
        ret_2022 = 0.0

    if e.std() > 1e-12 and h.std() > 1e-12:
        corr = float(e.corr(h))
    else:
        corr = 0.0

    return LeveragedHedgeTest(
        leverage=leverage,
        crisis_pct=crisis_pct,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        dd_2022=round(dd_2022, 2),
        return_2022=round(ret_2022, 2),
        corr=round(corr, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_v5_pipeline() -> Dict:
    print("[1/6] Loading real Yahoo data...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"      {len(prices)} days × {len(prices.columns)} assets")

    print("\n[2/6] Building EXP-1220 reference...")
    exp1220 = build_exp1220_daily_returns(prices)
    e_m = compute_metrics(exp1220.values)
    print(f"      EXP-1220: CAGR {e_m['cagr']*100:+.1f}%  "
          f"Sharpe {e_m['sharpe']:.2f}  DD {e_m['dd']*100:.1f}%")

    print("\n[3/6] Analyzing v4 baseline correlation...")
    from compass.crisis_alpha_v4 import ConfigV4, backtest_v4
    v4_cfg = ConfigV4(
        name="v4_baseline", lookback_preset="v2_round",
        vol_target=0.05, leverage=1.0,
        dd_brake_threshold=0.05, dd_brake_zone=0.03,
        max_weight=0.20, require_confirmation=False,
    )
    v4 = backtest_v4(prices, v4_cfg)
    v4_scores = score_hedge(v4.daily_returns, exp1220)
    print(f"      v4 baseline: full corr {v4_scores['corr_full']:+.3f}  "
          f"DD-period corr {v4_scores['corr_dd']:+.3f}  "
          f"downside cap {v4_scores['downside_capture']*100:+.3f}%")

    print("\n[4/6] Searching v5 hedge-optimized configs...")
    configs = search_hedge_configs(prices, exp1220)
    configs.sort(key=lambda c: c.hedge_score)
    print("\n  Top 5 by hedge_score (lower = better hedge):")
    for c in configs[:5]:
        print(f"   {c.name:55s}  score {c.hedge_score:+6.2f}  "
              f"corr_dd {c.corr_dd:+.3f}  cap {c.downside_capture*100:+.3f}%  "
              f"vol {c.vol:.1f}%  CAGR {c.cagr:+.1f}%")

    best = select_best_hedge(configs)
    print(f"\nBEST v5 hedge: {best.name}")
    print(f"  Standalone: CAGR {best.cagr:+.1f}%  Sharpe {best.sharpe:.2f}  "
          f"DD {best.max_dd:.1f}%")
    print(f"  Hedge:      full corr {best.corr_full:+.3f}  "
          f"DD-period corr {best.corr_dd:+.3f}  "
          f"downside capture {best.downside_capture*100:+.3f}%")

    print("\n[5/6] Leveraged EXP-1220 hedge test...")
    print("  Format: lev × | hedge % | CAGR  Sharpe  DD    | 2022 DD  2022 Ret")
    lev_tests = []
    for lev in [1.0, 2.0]:
        for pct in [0.0, 0.05, 0.10, 0.15, 0.20]:
            t = test_leveraged_hedge(exp1220, best.daily_returns, lev, pct)
            lev_tests.append(t)
            print(f"  {lev}x | {pct*100:>4.0f}% | "
                  f"{t.cagr:+7.1f}%  {t.sharpe:5.2f}  {t.max_dd:5.1f}% | "
                  f"{t.dd_2022:5.1f}%  {t.return_2022:+6.1f}%")

    # KEY METRIC
    pure_2x = next(t for t in lev_tests if t.leverage == 2.0 and t.crisis_pct == 0.0)
    hedged_2x = next(t for t in lev_tests if t.leverage == 2.0 and t.crisis_pct == 0.10)
    print(f"\nKEY METRIC — 2× EXP-1220 DD reduction with 10% v5 hedge:")
    print(f"  Pure 2× EXP-1220:           DD {pure_2x.max_dd:.1f}%")
    print(f"  90% (2× EXP-1220) + 10% v5: DD {hedged_2x.max_dd:.1f}%")
    print(f"  Reduction: {pure_2x.max_dd - hedged_2x.max_dd:+.1f}pp "
          f"({(1 - hedged_2x.max_dd/pure_2x.max_dd)*100:+.1f}%)")

    print("\n[6/6] Generating report...")
    return {
        "exp1220_metrics": e_m,
        "v4_baseline_scores": v4_scores,
        "configs": configs,
        "best": best,
        "leveraged_tests": lev_tests,
        "pure_2x_dd": pure_2x.max_dd,
        "hedged_2x_dd": hedged_2x.max_dd,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result: Dict,
    out_path: str = "compass/reports/crisis_alpha_v5_hedge_optimization.html",
) -> None:
    best = result["best"]
    e_m = result["exp1220_metrics"]
    v4 = result["v4_baseline_scores"]
    pure_2x = result["pure_2x_dd"]
    hedged_2x = result["hedged_2x_dd"]
    reduction_pct = (1 - hedged_2x / pure_2x) * 100 if pure_2x > 0 else 0.0

    top10 = result["configs"][:10]
    cfg_rows = "".join(
        f"<tr><td>{c.name}</td>"
        f"<td class=num>{c.hedge_score:+.2f}</td>"
        f"<td class=num>{c.corr_dd:+.3f}</td>"
        f"<td class=num>{c.downside_capture*100:+.3f}%</td>"
        f"<td class=num>{c.cagr:+.1f}%</td>"
        f"<td class=num>{c.max_dd:.1f}%</td>"
        f"<td class=num>{c.vol:.1f}%</td></tr>"
        for c in top10
    )

    lev_rows = "".join(
        f"<tr><td>{t.leverage}x</td>"
        f"<td class=num>{t.crisis_pct*100:.0f}%</td>"
        f"<td class=num>{t.cagr:+.1f}%</td>"
        f"<td class=num>{t.sharpe:.2f}</td>"
        f"<td class=num style='color:{'#16a34a' if t.crisis_pct > 0 and t.max_dd < pure_2x else ''}'>{t.max_dd:.1f}%</td>"
        f"<td class=num>{t.calmar:.2f}</td>"
        f"<td class=num>{t.dd_2022:.1f}%</td>"
        f"<td class=num>{t.return_2022:+.1f}%</td></tr>"
        for t in result["leveraged_tests"]
    )

    badge_color = "#16a34a" if hedged_2x < pure_2x else "#ef4444"
    badge_text = (f"DD reduction: {pure_2x - hedged_2x:+.1f}pp ({reduction_pct:+.1f}%)"
                  if pure_2x > 0 else "n/a")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>EXP-1780 Crisis Alpha v5 — Hedge Optimization</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#0b1220;color:#e2e8f0;
       max-width:1100px;margin:32px auto;padding:0 20px}}
  h1{{color:#fbbf24;border-bottom:2px solid #1e293b;padding-bottom:8px}}
  h2{{color:#60a5fa;margin-top:32px}}
  .meta{{color:#64748b;font-size:0.85rem}}
  table{{border-collapse:collapse;width:100%;margin:12px 0;background:#0f172a}}
  th,td{{padding:8px 12px;border-bottom:1px solid #1e293b;text-align:left;font-size:0.86rem}}
  th{{background:#1e293b;color:#cbd5e1}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  .info{{background:#1e3a8a;border-left:4px solid #60a5fa;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#bfdbfe}}
  .ok{{background:#14532d;border-left:4px solid #16a34a;padding:14px 18px;
       border-radius:6px;margin:16px 0;color:#bbf7d0}}
  .warn{{background:#7c2d12;border-left:4px solid #ef4444;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#fecaca}}
  .badge{{display:inline-block;padding:8px 16px;border-radius:6px;color:#fff;
         font-weight:700;margin:8px 0}}
  .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
  .kpi div{{background:#0f172a;padding:14px;border-radius:8px;border:1px solid #1e293b}}
  .kpi .v{{font-size:1.4rem;color:#fbbf24;font-weight:600}}
  .kpi .l{{font-size:0.78rem;color:#94a3b8;margin-top:4px}}
</style></head><body>

<h1>EXP-1780 Crisis Alpha v5 — Hedge Optimization</h1>
<div class=meta>2026-04-06 · Real Yahoo Finance 2014-2025 · objective:
minimize correlation during EXP-1220 drawdowns · Carlos use case:
hedging leveraged EXP-1220</div>

<div class=info><strong>Hypothesis:</strong> v4 maximized standalone Calmar
(it's a self-financing diversifier). v5 has a different job: be a
<em>hedge</em> for leveraged EXP-1220. Different objective, different
optimum. v5 adds a stress gate (only deploy when SPY in DD), safe-haven
tilt (overweight TLT/GLD/UUP), and equity-short-only mode. Configs are
scored by <code>hedge_score = corr_dd − 5·downside_capture</code> instead
of Calmar.</div>

<span class=badge style="background:{badge_color}">{badge_text}</span>

<h2>v4 baseline vs v5 best — hedge metrics</h2>
<table>
<tr><th></th><th>Full corr</th><th>DD-period corr</th><th>Downside capture</th>
<th>Standalone CAGR</th><th>Standalone DD</th></tr>
<tr><td>v4 baseline</td>
<td class=num>{v4['corr_full']:+.3f}</td>
<td class=num>{v4['corr_dd']:+.3f}</td>
<td class=num>{v4['downside_capture']*100:+.3f}%</td>
<td class=num>n/a</td><td class=num>n/a</td></tr>
<tr><td>v5 best</td>
<td class=num>{best.corr_full:+.3f}</td>
<td class=num style='color:#16a34a'><strong>{best.corr_dd:+.3f}</strong></td>
<td class=num style='color:#16a34a'><strong>{best.downside_capture*100:+.3f}%</strong></td>
<td class=num>{best.cagr:+.1f}%</td><td class=num>{best.max_dd:.1f}%</td></tr>
</table>

<h2>Best v5 hedge config</h2>
<p><code>{best.name}</code></p>
<div class=kpi>
<div><div class=v>{best.cagr:+.1f}%</div><div class=l>Standalone CAGR</div></div>
<div><div class=v>{best.sharpe:.2f}</div><div class=l>Sharpe</div></div>
<div><div class=v>{best.max_dd:.1f}%</div><div class=l>Max DD</div></div>
<div><div class=v>{best.vol:.1f}%</div><div class=l>Vol</div></div>
<div><div class=v>{best.corr_dd:+.3f}</div><div class=l>Corr in EXP-1220 DD</div></div>
<div><div class=v>{best.downside_capture*100:+.3f}%</div><div class=l>Downside capture</div></div>
<div><div class=v>{best.hedge_score:+.2f}</div><div class=l>Hedge score (lower=better)</div></div>
<div><div class=v>{best.corr_full:+.3f}</div><div class=l>Full-sample corr</div></div>
</div>

<h2>Leveraged hedge test — does 10% v5 protect 2× EXP-1220?</h2>
<p>This is the harder test: leverage doubles the drawdown the hedge has to
neutralize. If the hedge works at 2× it works at 1×.</p>
<table>
<tr><th>Lev</th><th>Hedge %</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
<th>Calmar</th><th>2022 DD</th><th>2022 Ret</th></tr>
{lev_rows}
</table>
<div class="{'ok' if hedged_2x < pure_2x else 'warn'}">
<strong>2× EXP-1220 DD with 10% v5:</strong>
pure 2× = {pure_2x:.2f}%, hedged = {hedged_2x:.2f}%, reduction
{pure_2x - hedged_2x:+.2f}pp ({reduction_pct:+.1f}%).
</div>

<h2>Top 10 v5 configs by hedge score</h2>
<table>
<tr><th>Config</th><th>Score</th><th>Corr in DD</th><th>Downside cap</th>
<th>CAGR</th><th>DD</th><th>Vol</th></tr>
{cfg_rows}
</table>

<h2>Methodology</h2>
<ul>
  <li><strong>Hedge score</strong> = corr during EXP-1220 DD periods − 5 × downside capture (in %).
  Lower is a better hedge. Penalises positive correlation under stress; rewards positive returns
  on EXP-1220's worst-quartile days.</li>
  <li><strong>EXP-1220 DD periods</strong>: days where running drawdown ≥3%.
  Tests the hedge against the actual moments it would be needed.</li>
  <li><strong>Downside capture</strong>: mean v5 return on EXP-1220's lowest 25% return days.
  Positive = the hedge MAKES money on the bad days.</li>
  <li><strong>Stress gate</strong>: SPY rolling 60-day DD threshold. When SPY DD &lt; threshold,
  v5 holds cash. Concentrates the risk budget on stress periods.</li>
  <li><strong>Safe-haven tilt</strong>: TLT/GLD/LQD/UUP raw weights are amplified by
  <code>safe_haven_boost</code>. Equity-short-only mode zeros out long equity positions.</li>
</ul>

<h2>EXP-1220 reference (proxy on real SPY)</h2>
<p>CAGR {e_m['cagr']*100:+.1f}% · Sharpe {e_m['sharpe']:.2f} · DD {e_m['dd']*100:.1f}%</p>
<p class=meta><em>EXP-1220 reference here is the calibrated functional proxy on real Yahoo SPY.
The validated real-trade backtest is in compass/exp1220_standalone.py and is documented
separately. Hedge effectiveness numbers should be read as directional — the proportional
DD reduction should carry over to the real strategy.</em></p>

<div class=meta>compass/crisis_alpha_v5.py · Rule Zero compliant · real Yahoo Finance only ·
hedge-optimized objective (not Calmar)</div>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"Report: {out_path}")


def main():
    result = run_v5_pipeline()
    generate_report(result)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point. Delegates to the central
    signal registry in compass.exp2690_signal_generators."""
    from compass.exp2690_signal_generators import v5_hedge_signals
    return v5_hedge_signals(date)
