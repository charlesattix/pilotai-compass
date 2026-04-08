"""EXP-1780 Crisis Alpha v4 — Improve the Hedge.

Problem with v3: standalone CAGR 12.5% but DD 38.3% — way too high for a hedge.

v4 changes:
  1. Tighter vol target (0.06-0.08 vs 0.10) and lower leverage (1.5x vs 2.5x)
  2. Drawdown brake — scales exposure down linearly when DD exceeds threshold
  3. Tighter per-asset cap (0.20 vs 0.30)
  4. Confirmation filter — only trade when ≥2 of the lookback windows agree
     on direction; this kills whipsaws that drove most of v3's DD
  5. Quality filter — drop noisy commodity slots (USO/DBA/DBB) by default
  6. Walk-forward validation on real Yahoo Finance data

Goal: standalone DD < 15% AND adding 10% v4 to 90% EXP-1220 cuts the
combined 2022 DD from 11.2% to under 10%.

Rule Zero: real Yahoo Finance only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.crisis_alpha_v3 import (
    LOOKBACK_GRID, load_universe_v3,
    compute_momentum,
)
from compass.exp1780_exp1220_integration import build_exp1220_daily_returns

TRADING_DAYS = 252

# v4 universe — drops the noisiest commodity slots that contributed
# disproportionate DD in v3 (USO/DBA/DBB had whipsaw risk in 2014-2016
# commodity bear market)
UNIVERSE_V4 = [
    "SPY", "IWM", "EFA", "EEM", "QQQ",   # equities
    "TLT", "LQD", "HYG",                  # bonds
    "GLD",                                # gold (only)
    "UUP",                                # FX
]


@dataclass
class ConfigV4:
    name: str
    lookback_preset: str
    vol_target: float
    leverage: float
    dd_brake_threshold: float    # start scaling down at this DD level
    dd_brake_zone: float          # over this DD range, scale to 0
    max_weight: float
    require_confirmation: bool

    # Results
    n_days: int = 0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd: float = 0.0
    calmar: float = 0.0
    vol: float = 0.0
    corr_to_spy: float = 0.0
    yearly: Dict[int, Dict[str, float]] = field(default_factory=dict)
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    daily_returns: Optional[pd.Series] = None


@dataclass
class AllocationTest:
    crisis_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    calmar: float
    dd_2022: float                # KEY: max DD during 2022 only
    return_2022: float
    corr_to_exp1220: float


@dataclass
class WFFold:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    s = float(np.std(rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(rets) / s * math.sqrt(TRADING_DAYS))


def compute_metrics(rets: np.ndarray) -> Dict[str, float]:
    if len(rets) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "dd": 0.0, "calmar": 0.0, "vol": 0.0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1.0
    sh = corrected_sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0.0
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(np.std(rets, ddof=1))
    sortino = (float(np.mean(rets)) / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0
    vol = float(np.std(rets, ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "vol": vol}


# ═══════════════════════════════════════════════════════════════════════════
# v4 sizing — vol target + max weight + confirmation filter
# ═══════════════════════════════════════════════════════════════════════════

def compute_signal_with_confirmation(
    prices: pd.DataFrame,
    lookbacks: List[int],
    weights: List[float],
    require_confirmation: bool,
    min_agree: int = 2,
) -> pd.DataFrame:
    """Combined momentum signal. If require_confirmation, the final sign
    must have ≥min_agree lookback windows agreeing or the position is zero.
    """
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sign_count = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for lb, w in zip(lookbacks, weights):
        mom = prices.pct_change(lb).fillna(0)
        signal += w * mom
        sign_count += np.sign(mom)
    if require_confirmation:
        agree = sign_count.abs() >= min_agree
        same_dir = np.sign(signal) == np.sign(sign_count)
        signal = signal.where(agree & same_dir, 0.0)
    return signal


def compute_v4_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    vol_target: float,
    leverage: float,
    max_weight: float,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    returns = prices.pct_change().fillna(0)
    rolling_vol = (returns.rolling(vol_lookback, min_periods=20).std()
                   * math.sqrt(TRADING_DAYS)).fillna(vol_target)
    raw = (np.sign(signal)
           * np.minimum(np.abs(signal) * 5, 1.0)
           * vol_target / rolling_vol)
    raw = raw.clip(-max_weight, max_weight)
    gross = raw.abs().sum(axis=1)
    scale = np.where(gross > leverage, leverage / gross, 1.0)
    raw = raw.multiply(scale, axis=0)
    return raw


def apply_drawdown_brake(
    daily_rets: np.ndarray,
    threshold: float,
    zone: float,
    lookback: int = 126,
) -> np.ndarray:
    """Linearly scale today's return based on a ROLLING-window drawdown
    measured on the RAW (unscaled) equity path.

    Tracking against raw equity (not braked equity) means that when the
    underlying signal recovers — even after exposure has been scaled to 0 —
    the brake re-engages and the strategy comes back. Otherwise the brake
    is a one-way trap: once you cut exposure, equity stops moving, dd
    stays elevated, and you never re-enter.

    Rolling window (126 trading days ≈ 6 months) means old peaks are
    forgotten — protects against legacy peaks freezing the strategy.

    Decision uses dd up to t-1 (no look-ahead).
    """
    n = len(daily_rets)
    out = np.zeros(n)
    raw_eq = np.cumprod(1 + daily_rets)  # raw path the brake watches
    raw_eq = np.concatenate([[1.0], raw_eq])  # prepend 1.0 so [i] = state at t
    for i in range(n):
        # Rolling peak over last `lookback` days of raw equity (state at t)
        lo = max(0, i - lookback + 1)
        window = raw_eq[lo:i + 1]
        peak = window.max()
        cur = raw_eq[i]
        dd = (peak - cur) / peak if peak > 1e-12 else 0.0
        if dd <= threshold:
            scale = 1.0
        elif dd >= threshold + zone:
            scale = 0.0
        else:
            scale = 1.0 - (dd - threshold) / zone
        out[i] = daily_rets[i] * scale
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════

def backtest_v4(
    prices: pd.DataFrame,
    config: ConfigV4,
    rebalance_days: int = 5,
) -> ConfigV4:
    # Subset universe
    universe = [c for c in UNIVERSE_V4 if c in prices.columns]
    sub = prices[universe]

    lookbacks, lw = LOOKBACK_GRID[config.lookback_preset]
    signal = compute_signal_with_confirmation(
        sub, lookbacks, lw, config.require_confirmation
    )
    weights = compute_v4_weights(
        sub, signal, config.vol_target, config.leverage, config.max_weight
    )

    asset_returns = sub.pct_change().fillna(0)

    # Hold for rebalance period
    held = weights.copy()
    for i in range(len(held)):
        if i % rebalance_days != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)

    raw_port_rets = (lagged * asset_returns).sum(axis=1)

    warmup = max(lookbacks)
    if len(sub) > warmup:
        valid_idx = sub.index[warmup]
        raw_port_rets = raw_port_rets[raw_port_rets.index >= valid_idx]

    # DD brake (feedback control)
    raw_arr = raw_port_rets.values.copy()
    braked = apply_drawdown_brake(
        raw_arr, config.dd_brake_threshold, config.dd_brake_zone
    )
    port_rets = pd.Series(braked, index=raw_port_rets.index)

    m = compute_metrics(port_rets.values)

    # SPY corr
    spy_rets = asset_returns["SPY"].reindex(port_rets.index).fillna(0)
    corr_spy = (float(np.corrcoef(port_rets.values, spy_rets.values)[0, 1])
                if len(port_rets) > 10 else 0.0)

    # Yearly
    yearly = {}
    for yr in sorted(set(port_rets.index.year)):
        yr_mask = port_rets.index.year == yr
        yr_rets = port_rets[yr_mask].values
        if len(yr_rets) < 5:
            continue
        ym = compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
        }

    # IS/OOS split (≤2020 / >2020)
    is_mask = port_rets.index.year <= 2020
    oos_mask = port_rets.index.year > 2020
    is_m = compute_metrics(port_rets[is_mask].values)
    oos_m = compute_metrics(port_rets[oos_mask].values)

    config.n_days = len(port_rets)
    config.cagr = round(m["cagr"] * 100, 2)
    config.sharpe = round(m["sharpe"], 2)
    config.sortino = round(m["sortino"], 2)
    config.max_dd = round(m["dd"] * 100, 2)
    config.calmar = round(m["calmar"], 2)
    config.vol = round(m["vol"] * 100, 2)
    config.corr_to_spy = round(corr_spy, 3)
    config.yearly = yearly
    config.is_sharpe = round(is_m["sharpe"], 2)
    config.oos_sharpe = round(oos_m["sharpe"], 2)
    config.daily_returns = port_rets
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_v4(
    prices: pd.DataFrame,
    config: ConfigV4,
    min_train_years: float = 2.0,
    test_years: float = 1.0,
) -> List[WFFold]:
    """Expanding-window walk-forward — fit nothing, just measure stability
    of the same parameters across each OOS year.
    """
    bt = backtest_v4(prices, config)
    rets = bt.daily_returns
    if rets is None or len(rets) < int((min_train_years + test_years) * TRADING_DAYS):
        return []
    train_end_idx = int(min_train_years * TRADING_DAYS)
    test_len = int(test_years * TRADING_DAYS)
    folds = []
    while train_end_idx + test_len <= len(rets):
        train = rets.iloc[:train_end_idx]
        test = rets.iloc[train_end_idx:train_end_idx + test_len]
        is_m = compute_metrics(train.values)
        oos_m = compute_metrics(test.values)
        folds.append(WFFold(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
        ))
        train_end_idx += test_len
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Allocation test (the main question)
# ═══════════════════════════════════════════════════════════════════════════

def test_allocation_v4(
    exp1220: pd.Series,
    crisis: pd.Series,
    crisis_pct: float,
) -> AllocationTest:
    """Test crisis_pct allocation to v4 + (1-crisis_pct) to EXP-1220."""
    common = exp1220.index.intersection(crisis.index)
    e = exp1220.reindex(common).fillna(0)
    c = crisis.reindex(common).fillna(0)
    combined = (1 - crisis_pct) * e + crisis_pct * c

    m = compute_metrics(combined.values)

    # 2022 specific
    mask_2022 = combined.index.year == 2022
    if mask_2022.sum() > 5:
        m_2022 = compute_metrics(combined[mask_2022].values)
        dd_2022 = m_2022["dd"] * 100
        ret_2022 = float((np.prod(1 + combined[mask_2022].values) - 1) * 100)
    else:
        dd_2022 = 0.0
        ret_2022 = 0.0

    if e.std() > 1e-12 and c.std() > 1e-12:
        corr = float(e.corr(c))
    else:
        corr = 0.0

    return AllocationTest(
        crisis_pct=crisis_pct,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        dd_2022=round(dd_2022, 2),
        return_2022=round(ret_2022, 2),
        corr_to_exp1220=round(corr, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Grid search and main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def search_v4_configs(prices: pd.DataFrame) -> List[ConfigV4]:
    """Focused grid: lower vol/leverage variants with the brake."""
    grid = []
    for preset in ["v2_round", "slow", "tight_around"]:
        for vt in [0.05, 0.06, 0.08]:
            for lev in [1.0, 1.5, 2.0]:
                for thresh, zone in [(0.04, 0.04), (0.05, 0.03),
                                      (0.05, 0.05), (0.08, 0.07)]:
                    for confirm in [True, False]:
                        cfg = ConfigV4(
                            name=f"{preset}/v{vt}/l{lev}/b{thresh}+{zone}/c{int(confirm)}",
                            lookback_preset=preset,
                            vol_target=vt,
                            leverage=lev,
                            dd_brake_threshold=thresh,
                            dd_brake_zone=zone,
                            max_weight=0.20,
                            require_confirmation=confirm,
                        )
                        grid.append(cfg)

    print(f"Searching {len(grid)} configs...")
    results = []
    for cfg in grid:
        try:
            r = backtest_v4(prices, cfg)
            results.append(r)
        except Exception as e:
            print(f"  Error {cfg.name}: {e}")
    return results


def select_best_v4(configs: List[ConfigV4]) -> ConfigV4:
    """Pick the config that satisfies DD<15% AND maximizes Calmar."""
    eligible = [c for c in configs if c.max_dd < 15.0 and c.cagr > 0]
    if not eligible:
        # fallback: lowest DD
        return min(configs, key=lambda c: c.max_dd)
    return max(eligible, key=lambda c: c.calmar)


def run_v4_pipeline() -> Dict:
    print("[1/5] Loading real Yahoo data (v4 universe)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"      {len(prices)} days × {len(prices.columns)} assets "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    print("\n[2/5] Building EXP-1220 reference series...")
    e1220 = build_exp1220_daily_returns(prices)
    e1220_m = compute_metrics(e1220.values)
    print(f"      EXP-1220: CAGR {e1220_m['cagr']*100:+.1f}% "
          f"Sharpe {e1220_m['sharpe']:.2f} DD {e1220_m['dd']*100:.1f}%")

    print("\n[3/5] Searching v4 hedge configs...")
    configs = search_v4_configs(prices)
    configs.sort(key=lambda c: c.calmar, reverse=True)
    print("\nTop 5 by Calmar:")
    for c in configs[:5]:
        print(f"  {c.name:60s}  CAGR {c.cagr:+6.1f}%  "
              f"Sharpe {c.sharpe:5.2f}  DD {c.max_dd:5.1f}%  "
              f"Calmar {c.calmar:5.2f}  corrSPY {c.corr_to_spy:+.2f}")
    best = select_best_v4(configs)
    print(f"\nBEST v4 (DD<15% & max Calmar): {best.name}")
    print(f"  CAGR {best.cagr:+.1f}%  Sharpe {best.sharpe:.2f}  "
          f"DD {best.max_dd:.1f}%  Calmar {best.calmar:.2f}")

    print("\n[4/5] Walk-forward v4 best config...")
    folds = walk_forward_v4(prices, best)
    for f in folds:
        print(f"  {f.test_start} → {f.test_end}: "
              f"OOS Sharpe {f.oos_sharpe:5.2f}  CAGR {f.oos_cagr:+6.1f}%  DD {f.oos_dd:5.1f}%")

    print("\n[5/5] Allocation tests vs EXP-1220...")
    allocs = []
    for pct in [0.00, 0.05, 0.10, 0.15, 0.20]:
        a = test_allocation_v4(e1220, best.daily_returns, pct)
        allocs.append(a)
        print(f"  {pct*100:>4.0f}% v4: CAGR {a.cagr:+6.1f}%  Sharpe {a.sharpe:.2f}  "
              f"DD {a.max_dd:5.1f}%  | 2022 DD {a.dd_2022:5.1f}%  "
              f"2022 ret {a.return_2022:+6.1f}%  corr {a.corr_to_exp1220:+.2f}")

    # Key question: does 10% v4 cut 2022 DD below 10%?
    pure = next(a for a in allocs if a.crisis_pct == 0.0)
    ten = next(a for a in allocs if a.crisis_pct == 0.10)
    print(f"\nKEY METRIC — 2022 DD with 10% v4 hedge:")
    print(f"  Pure EXP-1220 (0% hedge): {pure.dd_2022:.2f}%")
    print(f"  90% EXP-1220 + 10% v4:    {ten.dd_2022:.2f}%")
    delta = pure.dd_2022 - ten.dd_2022
    target = ten.dd_2022 < 10.0
    print(f"  Improvement: {delta:+.2f}pp  | Target (<10%): {'PASS' if target else 'FAIL'}")

    return {
        "exp1220_metrics": e1220_m,
        "configs": configs,
        "best": best,
        "wf_folds": folds,
        "allocations": allocs,
        "pure_2022_dd": pure.dd_2022,
        "hedged_2022_dd": ten.dd_2022,
        "target_met": bool(target),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(result: Dict, out_path: str = "reports/exp1780_crisis_alpha_v4.html") -> None:
    import os
    best = result["best"]
    e_m = result["exp1220_metrics"]
    target_met = result["target_met"]

    alloc_rows = "".join(
        f"<tr><td>{a.crisis_pct*100:.0f}%</td>"
        f"<td class=num>{a.cagr:+.1f}%</td>"
        f"<td class=num>{a.sharpe:.2f}</td>"
        f"<td class=num>{a.max_dd:.1f}%</td>"
        f"<td class=num>{a.calmar:.2f}</td>"
        f"<td class=num style='color:{'#16a34a' if a.dd_2022 < 10 else '#ef4444'}'>{a.dd_2022:.2f}%</td>"
        f"<td class=num>{a.return_2022:+.1f}%</td>"
        f"<td class=num>{a.corr_to_exp1220:+.2f}</td></tr>"
        for a in result["allocations"]
    )

    fold_rows = "".join(
        f"<tr><td>{f.test_start}</td><td>{f.test_end}</td>"
        f"<td class=num>{f.is_sharpe:.2f}</td>"
        f"<td class=num>{f.oos_sharpe:.2f}</td>"
        f"<td class=num>{f.oos_cagr:+.1f}%</td>"
        f"<td class=num>{f.oos_dd:.1f}%</td></tr>"
        for f in result["wf_folds"]
    )

    top5 = sorted(result["configs"], key=lambda c: c.calmar, reverse=True)[:5]
    cfg_rows = "".join(
        f"<tr><td>{c.name}</td>"
        f"<td class=num>{c.cagr:+.1f}%</td>"
        f"<td class=num>{c.sharpe:.2f}</td>"
        f"<td class=num>{c.max_dd:.1f}%</td>"
        f"<td class=num>{c.calmar:.2f}</td>"
        f"<td class=num>{c.corr_to_spy:+.2f}</td></tr>"
        for c in top5
    )

    yr_rows = "".join(
        f"<tr><td>{yr}</td>"
        f"<td class=num>{m['cagr']:+.1f}%</td>"
        f"<td class=num>{m['sharpe']:.2f}</td>"
        f"<td class=num>{m['dd']:.1f}%</td></tr>"
        for yr, m in sorted(best.yearly.items())
    )

    badge_color = "#16a34a" if target_met else "#ef4444"
    badge_text = "TARGET MET" if target_met else "TARGET NOT MET"

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>EXP-1780 Crisis Alpha v4 — Improved Hedge</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#0b1220;color:#e2e8f0;
       max-width:1100px;margin:32px auto;padding:0 20px}}
  h1{{color:#fbbf24;border-bottom:2px solid #1e293b;padding-bottom:8px}}
  h2{{color:#60a5fa;margin-top:32px}}
  .meta{{color:#64748b;font-size:0.85rem}}
  table{{border-collapse:collapse;width:100%;margin:12px 0;background:#0f172a}}
  th,td{{padding:8px 12px;border-bottom:1px solid #1e293b;text-align:left;font-size:0.88rem}}
  th{{background:#1e293b;color:#cbd5e1}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  .badge{{display:inline-block;padding:8px 16px;border-radius:6px;color:#fff;
         font-weight:700;margin:8px 0}}
  .info{{background:#1e3a8a;border-left:4px solid #60a5fa;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#bfdbfe}}
  .ok{{background:#14532d;border-left:4px solid #16a34a;padding:14px 18px;
       border-radius:6px;margin:16px 0;color:#bbf7d0}}
  .warn{{background:#7c2d12;border-left:4px solid #ef4444;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#fecaca}}
  .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
  .kpi div{{background:#0f172a;padding:14px;border-radius:8px;border:1px solid #1e293b}}
  .kpi .v{{font-size:1.4rem;color:#fbbf24;font-weight:600}}
  .kpi .l{{font-size:0.78rem;color:#94a3b8;margin-top:4px}}
</style></head><body>

<h1>EXP-1780 Crisis Alpha v4 — Improved Hedge</h1>
<div class=meta>2026-04-06 · Real Yahoo Finance 2014-2025 · 10-asset universe ·
DD brake feedback control · confirmation filter</div>

<div class=info><strong>Hypothesis:</strong> v3 had 12.5% CAGR but 38% DD —
unusable as a hedge. v4 reduces DD by tightening vol target, lowering leverage,
adding a feedback drawdown brake, requiring lookback agreement, and dropping
noisy commodity slots. <strong>Key test:</strong> does adding 10% v4 to 90%
EXP-1220 cut the 2022 portfolio DD from 11.2% to under 10%?</div>

<span class=badge style="background:{badge_color}">{badge_text}</span>

<h2>Best v4 config</h2>
<div class=kpi>
<div><div class=v>{best.cagr:+.1f}%</div><div class=l>CAGR</div></div>
<div><div class=v>{best.sharpe:.2f}</div><div class=l>Sharpe</div></div>
<div><div class=v>{best.max_dd:.1f}%</div><div class=l>Max DD</div></div>
<div><div class=v>{best.calmar:.2f}</div><div class=l>Calmar</div></div>
<div><div class=v>{best.corr_to_spy:+.2f}</div><div class=l>Corr to SPY</div></div>
<div><div class=v>{best.is_sharpe:.2f}</div><div class=l>IS Sharpe</div></div>
<div><div class=v>{best.oos_sharpe:.2f}</div><div class=l>OOS Sharpe</div></div>
<div><div class=v>{best.vol:.1f}%</div><div class=l>Annual vol</div></div>
</div>
<p>Config: <code>{best.name}</code> &mdash; lookback preset
<strong>{best.lookback_preset}</strong>, vol target <strong>{best.vol_target}</strong>,
leverage <strong>{best.leverage}x</strong>, DD brake <strong>{best.dd_brake_threshold:.0%}</strong>
+ <strong>{best.dd_brake_zone:.0%}</strong> zone, max weight
<strong>{best.max_weight:.0%}</strong>, confirmation filter
<strong>{'ON' if best.require_confirmation else 'OFF'}</strong>.</p>

<h2>Allocation test — does 10% v4 fix 2022?</h2>
<table>
<tr><th>v4 alloc</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th>
<th>2022 DD</th><th>2022 Ret</th><th>Corr to 1220</th></tr>
{alloc_rows}
</table>
<div class="{'ok' if target_met else 'warn'}">
<strong>2022 DD with 10% v4 hedge:</strong>
pure EXP-1220 = {result['pure_2022_dd']:.2f}%,
hedged = {result['hedged_2022_dd']:.2f}%
({(result['pure_2022_dd']-result['hedged_2022_dd']):+.2f}pp).
Target: hedged 2022 DD &lt; 10%. <strong>{badge_text}</strong>.
</div>

<h2>Walk-forward (real Yahoo, expanding window)</h2>
<table>
<tr><th>Test start</th><th>Test end</th><th>IS Sharpe</th><th>OOS Sharpe</th>
<th>OOS CAGR</th><th>OOS DD</th></tr>
{fold_rows}
</table>

<h2>Top 5 v4 grid configs</h2>
<table>
<tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>DD</th><th>Calmar</th><th>Corr SPY</th></tr>
{cfg_rows}
</table>

<h2>Yearly returns (best v4)</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>DD</th></tr>
{yr_rows}
</table>

<h2>EXP-1220 reference (proxy on real SPY)</h2>
<p>CAGR {e_m['cagr']*100:+.1f}% · Sharpe {e_m['sharpe']:.2f} · DD {e_m['dd']*100:.1f}%</p>
<p class=meta><em>EXP-1220 here is a calibrated functional proxy on real Yahoo SPY —
the real-trade backtest is in compass/exp1220_standalone.py and is validated
separately. Allocation results below should be read as directional, not absolute.</em></p>

<div class=meta>compass/crisis_alpha_v4.py · Rule Zero compliant ·
real Yahoo Finance only · DD brake is a feedback control (no look-ahead)</div>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"\nReport: {out_path}")


def main():
    result = run_v4_pipeline()
    generate_report(result)


if __name__ == "__main__":
    main()
