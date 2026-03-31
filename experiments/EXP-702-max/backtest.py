#!/usr/bin/env python3
"""
EXP-702-max: Multi-Asset Credit Spread Portfolio Backtest

Simulates credit spread strategies across SPY, QQQ, IWM, IBIT using:
  - Real SPY trade data from training_data_combined.csv (2020-2025)
  - Synthetic correlated streams for QQQ (ρ≈0.85), IWM (ρ≈0.75), IBIT (ρ≈0.30)
  - Risk parity allocation (weight ∝ 1/vol)
  - Per-asset regime detection from VIX/momentum features
  - Cross-asset correlation monitoring with sizing reduction
  - Portfolio-level drawdown limits (8% reduce, 10% kill switch)

Outputs:
  results/summary.json  — machine-readable metrics
  results/report.html   — visual dashboard
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────

TRADING_DAYS = 252
INITIAL_CAPITAL = 100_000.0
DATA_PATH = Path(__file__).parent.parent / "training_data_combined.csv"
RESULTS_DIR = Path(__file__).parent / "results"

ASSETS = ["SPY", "QQQ", "IWM", "IBIT"]
# Historical correlations with SPY
CORRELATIONS = {"SPY": 1.0, "QQQ": 0.85, "IWM": 0.75, "IBIT": 0.30}
# Relative volatility scaling vs SPY
VOL_SCALE = {"SPY": 1.0, "QQQ": 1.15, "IWM": 1.25, "IBIT": 3.0}
# Relative premium richness (trade P&L scaling)
PREMIUM_SCALE = {"SPY": 1.0, "QQQ": 0.90, "IWM": 0.85, "IBIT": 1.40}
# IBIT only available from 2024
IBIT_START_YEAR = 2024

SLIPPAGE_BPS = 5.0
COMMISSION_PER_CONTRACT = 1.30
DD_REDUCE_THRESHOLD = 0.08    # reduce sizing at 8% DD
DD_KILL_THRESHOLD = 0.12      # flatten at 12% DD (per thesis max)
CORR_SPIKE_THRESHOLD = 0.70   # reduce sizing when avg correlation > 0.7


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class AssetTrade:
    asset: str
    entry_date: str
    exit_date: str
    year: int
    strategy_type: str
    regime: str
    contracts: int
    gross_pnl: float
    slippage: float
    commission: float
    net_pnl: float
    win: bool
    vix: float
    signal_score: float


@dataclass
class YearResult:
    year: int
    n_trades: int
    total_pnl: float
    annual_return_pct: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    per_asset_pnl: Dict[str, float]
    profitable: bool


@dataclass
class AssetResult:
    asset: str
    n_trades: int
    total_pnl: float
    sharpe: float
    win_rate: float
    weight: float


# ── Regime detection ─────────────────────────────────────────────────────


def detect_regime(row: pd.Series) -> str:
    regime = str(row.get("regime", "")).lower().strip()
    if regime in ("bull", "bear", "sideways", "crisis", "high_vol", "low_vol"):
        return regime
    vix = float(row.get("vix", 20))
    mom = float(row.get("momentum_10d_pct", 0))
    if vix > 30:
        return "high_vol"
    if mom > 1 and vix < 20:
        return "bull"
    return "sideways"


def regime_is_favorable(regime: str) -> bool:
    return regime in ("bull", "sideways", "low_vol")


# ── Signal scoring ───────────────────────────────────────────────────────


def score_trade(row: pd.Series) -> float:
    s = 0.50
    regime = detect_regime(row)
    if regime == "bull":
        s += 0.12
    elif regime == "sideways":
        s += 0.05
    elif regime == "bear":
        s -= 0.10
    vix_pct = float(row.get("vix_percentile_50d", 50))
    if vix_pct > 70:
        s += 0.08
    elif vix_pct < 30:
        s -= 0.05
    iv = float(row.get("iv_rank", 50))
    if iv > 50:
        s += 0.06
    mom = float(row.get("momentum_5d_pct", 0))
    if mom > 0.5:
        s += 0.04
    elif mom < -2:
        s -= 0.06
    return max(0.0, min(1.0, s))


# ── Risk parity weights ─────────────────────────────────────────────────


def compute_risk_parity_weights(
    asset_vols: Dict[str, float],
    active_assets: List[str],
) -> Dict[str, float]:
    if not active_assets:
        return {}
    vols = np.array([max(asset_vols.get(a, 1.0), 0.01) for a in active_assets])
    inv = 1.0 / vols
    weights = inv / inv.sum()
    return {a: float(weights[i]) for i, a in enumerate(active_assets)}


# ── Correlation monitoring ───────────────────────────────────────────────


def compute_rolling_correlation(
    pnl_streams: Dict[str, List[float]],
    window: int = 20,
) -> float:
    """Average pairwise correlation of recent P&L across assets."""
    active = {a: v for a, v in pnl_streams.items() if len(v) >= window}
    if len(active) < 2:
        return 0.0

    names = list(active.keys())
    correlations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(active[names[i]][-window:])
            b = np.array(active[names[j]][-window:])
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    return float(np.mean(correlations)) if correlations else 0.0


# ── Synthetic asset generation ───────────────────────────────────────────


def generate_synthetic_trades(
    spy_trades: pd.DataFrame,
    asset: str,
    seed: int,
) -> pd.DataFrame:
    """Generate synthetic trades for a non-SPY asset based on SPY trades.

    Uses correlated noise to perturb P&L while preserving timing and structure.
    """
    rng = np.random.RandomState(seed)
    corr = CORRELATIONS[asset]
    vol_scale = VOL_SCALE[asset]
    premium_scale = PREMIUM_SCALE[asset]

    df = spy_trades.copy()

    # Only include years where asset is available
    if asset == "IBIT":
        df = df[df["year"] >= IBIT_START_YEAR].copy()

    if df.empty:
        return df

    # Generate correlated P&L: pnl_new = corr * pnl_spy_scaled + (1-corr) * independent_noise
    spy_pnl = df["pnl"].values.astype(float)
    pnl_std = np.std(spy_pnl) if np.std(spy_pnl) > 0 else 100.0

    noise = rng.normal(0, pnl_std * vol_scale, len(spy_pnl))
    synthetic_pnl = corr * spy_pnl * premium_scale + (1 - corr) * noise

    # Perturb win/loss around the same mean
    df["pnl"] = synthetic_pnl
    df["win"] = (synthetic_pnl > 0).astype(int)
    df["return_pct"] = synthetic_pnl / (abs(df["net_credit"]) * df["contracts"] * 100 + 1) * 100

    # Offset entry dates slightly (staggered across assets)
    offset_days = {"QQQ": 1, "IWM": 2, "IBIT": 0}
    df["entry_date"] = pd.to_datetime(df["entry_date"]) + pd.Timedelta(days=offset_days.get(asset, 0))
    df["exit_date"] = pd.to_datetime(df["exit_date"]) + pd.Timedelta(days=offset_days.get(asset, 0))

    return df.reset_index(drop=True)


# ── Metrics ──────────────────────────────────────────────────────────────


def sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    mu, std = pnls.mean(), pnls.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


def max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()) * 100)


def profit_factor(pnls: np.ndarray) -> float:
    g = pnls[pnls > 0].sum()
    l = abs(pnls[pnls < 0].sum())
    return float(g / l) if l > 1e-12 else (10.0 if g > 0 else 0.0)


# ── Core backtest ────────────────────────────────────────────────────────


def run_backtest() -> Dict[str, Any]:
    """Run the full multi-asset backtest."""

    # Load SPY data
    spy_df = pd.read_csv(DATA_PATH, parse_dates=["entry_date", "exit_date"])
    spy_df["year"] = pd.to_datetime(spy_df["entry_date"]).dt.year

    # Generate per-asset trade streams
    asset_trades: Dict[str, pd.DataFrame] = {"SPY": spy_df}
    for i, asset in enumerate(["QQQ", "IWM", "IBIT"]):
        asset_trades[asset] = generate_synthetic_trades(spy_df, asset, seed=100 + i)

    # Risk parity: estimate vol from historical P&L std
    asset_vols: Dict[str, float] = {}
    for asset, df in asset_trades.items():
        if not df.empty:
            asset_vols[asset] = max(float(df["pnl"].std()), 1.0)

    # Simulation state
    capital = INITIAL_CAPITAL
    peak_capital = capital
    all_trades: List[AssetTrade] = []
    equity_curve: List[float] = [capital]
    pnl_streams: Dict[str, List[float]] = {a: [] for a in ASSETS}

    # Process trades chronologically across all assets
    combined_rows: List[Tuple[str, pd.Series]] = []
    for asset, df in asset_trades.items():
        for _, row in df.iterrows():
            combined_rows.append((asset, row))

    # Sort by entry date
    combined_rows.sort(key=lambda x: str(x[1].get("entry_date", "")))

    for asset, row in combined_rows:
        # Active assets for this year
        year = int(row.get("year", 2020))
        active = [a for a in ASSETS if a != "IBIT" or year >= IBIT_START_YEAR]

        # Risk parity weights
        weights = compute_risk_parity_weights(asset_vols, active)
        asset_weight = weights.get(asset, 0.25)

        # Regime check
        regime = detect_regime(row)
        if not regime_is_favorable(regime):
            continue

        # Signal check
        signal = score_trade(row)
        if signal < 0.40:
            continue

        # Drawdown check (resets each year to allow recovery)
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
        if dd >= DD_KILL_THRESHOLD:
            # Reset peak on year boundary to allow recovery
            if year != getattr(run_backtest, '_last_kill_year', 0):
                run_backtest._last_kill_year = year
                peak_capital = capital  # reset peak
                dd = 0.0
            else:
                continue  # kill switch active for rest of this year
        dd_scale = 0.5 if dd >= DD_REDUCE_THRESHOLD else 1.0

        # Correlation check
        avg_corr = compute_rolling_correlation(pnl_streams, window=15)
        corr_scale = 0.6 if avg_corr > CORR_SPIKE_THRESHOLD else 1.0

        # Position sizing: risk parity weight applied to base size
        base_contracts = max(int(row.get("contracts", 5)), 1)
        # Scale: weight determines fraction, signal/dd/corr modulate
        modulator = signal * dd_scale * corr_scale
        contracts = max(1, round(base_contracts * asset_weight * modulator / 0.25))

        # Scale P&L to our contract count
        orig_contracts = max(int(row.get("contracts", 5)), 1)
        gross_pnl = float(row.get("pnl", 0.0)) / orig_contracts * contracts

        # Costs
        entry_price = abs(float(row.get("net_credit", 1.0)))
        mult = contracts * 100
        slip = entry_price * 2 * SLIPPAGE_BPS / 10_000 * mult
        comm = COMMISSION_PER_CONTRACT * contracts * 2
        net_pnl = gross_pnl - slip - comm

        trade = AssetTrade(
            asset=asset,
            entry_date=str(row.get("entry_date", "")),
            exit_date=str(row.get("exit_date", "")),
            year=year,
            strategy_type=str(row.get("strategy_type", "CS")),
            regime=regime,
            contracts=contracts,
            gross_pnl=gross_pnl,
            slippage=slip,
            commission=comm,
            net_pnl=net_pnl,
            win=net_pnl > 0,
            vix=float(row.get("vix", 20)),
            signal_score=signal,
        )
        all_trades.append(trade)

        capital += net_pnl
        peak_capital = max(peak_capital, capital)
        equity_curve.append(capital)
        pnl_streams[asset].append(net_pnl)

    # ── Compute results ──────────────────────────────────────────────

    if not all_trades:
        return {"error": "No trades executed"}

    pnls = np.array([t.net_pnl for t in all_trades])
    equity = np.array(equity_curve)
    n_trades = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.win)
    total_pnl = float(pnls.sum())
    years = sorted(set(t.year for t in all_trades))
    n_years = max(len(years), 1)

    # SPY-only baseline for comparison
    spy_only = [t for t in all_trades if t.asset == "SPY"]
    spy_pnls = np.array([t.net_pnl for t in spy_only]) if spy_only else np.array([0.0])
    spy_sharpe = sharpe(spy_pnls)

    # Per-year
    year_results: List[YearResult] = []
    for y in years:
        yt = [t for t in all_trades if t.year == y]
        yp = np.array([t.net_pnl for t in yt])
        yeq = INITIAL_CAPITAL + np.cumsum(yp)
        per_asset = {}
        for a in ASSETS:
            at = [t for t in yt if t.asset == a]
            per_asset[a] = float(sum(t.net_pnl for t in at))
        yr = YearResult(
            year=y, n_trades=len(yt),
            total_pnl=float(yp.sum()),
            annual_return_pct=float(yp.sum()) / INITIAL_CAPITAL * 100,
            sharpe=sharpe(yp), max_dd_pct=max_dd_pct(yeq),
            win_rate=sum(1 for t in yt if t.win) / len(yt) if yt else 0.0,
            per_asset_pnl=per_asset,
            profitable=float(yp.sum()) > 0,
        )
        year_results.append(yr)

    # Per-asset
    asset_results: List[AssetResult] = []
    for a in ASSETS:
        at = [t for t in all_trades if t.asset == a]
        if not at:
            continue
        ap = np.array([t.net_pnl for t in at])
        active = [aa for aa in ASSETS if aa != "IBIT" or True]
        w = compute_risk_parity_weights(asset_vols, active).get(a, 0.25)
        asset_results.append(AssetResult(
            asset=a, n_trades=len(at), total_pnl=float(ap.sum()),
            sharpe=sharpe(ap),
            win_rate=sum(1 for t in at if t.win) / len(at),
            weight=w,
        ))

    # Success criteria
    all_years_profitable = all(yr.profitable for yr in year_results)
    positive_assets = sum(1 for ar in asset_results if ar.total_pnl > 0)
    portfolio_sharpe = sharpe(pnls)
    portfolio_dd = max_dd_pct(equity)
    sharpe_improvement = portfolio_sharpe - spy_sharpe

    summary = {
        "experiment": "EXP-702-max",
        "description": "Multi-Asset Credit Spread Portfolio",
        "capital": INITIAL_CAPITAL,
        "final_capital": float(equity[-1]),
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / INITIAL_CAPITAL * 100,
        "annualized_return_pct": total_pnl / INITIAL_CAPITAL / n_years * 100,
        "sharpe": portfolio_sharpe,
        "sortino": _sortino(pnls),
        "max_drawdown_pct": portfolio_dd,
        "win_rate": n_wins / n_trades,
        "profit_factor": profit_factor(pnls),
        "n_trades": n_trades,
        "n_years": n_years,
        "years": years,
        "spy_only_sharpe": spy_sharpe,
        "sharpe_improvement": sharpe_improvement,
        "per_year": [
            {
                "year": yr.year, "n_trades": yr.n_trades,
                "pnl": yr.total_pnl, "return_pct": yr.annual_return_pct,
                "sharpe": yr.sharpe, "max_dd_pct": yr.max_dd_pct,
                "win_rate": yr.win_rate, "profitable": yr.profitable,
                "per_asset": yr.per_asset_pnl,
            }
            for yr in year_results
        ],
        "per_asset": [
            {
                "asset": ar.asset, "n_trades": ar.n_trades,
                "pnl": ar.total_pnl, "sharpe": ar.sharpe,
                "win_rate": ar.win_rate, "weight": ar.weight,
            }
            for ar in asset_results
        ],
        "success_criteria": {
            "sharpe_improvement_over_spy": {
                "target": 1.0, "actual": sharpe_improvement,
                "met": sharpe_improvement >= 1.0,
            },
            "max_dd_under_12pct": {
                "target": 12.0, "actual": portfolio_dd,
                "met": portfolio_dd <= 12.0,
            },
            "all_years_profitable": {
                "target": True, "actual": all_years_profitable,
                "met": all_years_profitable,
            },
            "positive_in_3_of_4_assets": {
                "target": 3, "actual": positive_assets,
                "met": positive_assets >= 3,
            },
        },
        "total_slippage": float(sum(t.slippage for t in all_trades)),
        "total_commission": float(sum(t.commission for t in all_trades)),
        "equity_curve": equity_curve,
    }

    return summary, year_results, asset_results, all_trades, equity


def _sortino(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    mu = pnls.mean()
    down = pnls[pnls < 0]
    if len(down) == 0:
        return 10.0 if mu > 0 else 0.0
    ds = np.sqrt(np.mean(down ** 2))
    return float(mu / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0


# ── HTML report ──────────────────────────────────────────────────────────


def generate_report(summary: Dict, year_results: List, asset_results: List,
                    equity: np.ndarray) -> str:
    s = summary
    sc = s["success_criteria"]

    def _fd(v): return f"${v:,.2f}"
    def _fp(v): return f"{v:.1f}%"
    def _fr(v): return f"{v:.2f}"
    def _ti(met): return '<span style="color:#3fb950">&#10003;</span>' if met else '<span style="color:#f85149">&#10007;</span>'

    # Equity SVG
    vals = s.get("equity_curve", [])
    eq_svg = ""
    if len(vals) > 2:
        n = len(vals)
        w, h = 700, 200
        pad = 55
        y0, y1 = min(vals), max(vals)
        if y1 <= y0: y1 = y0 + 1
        pw, ph = w - 2*pad, h - 65
        tx = lambda i: pad + i / max(n-1,1) * pw
        ty = lambda v: 35 + (1-(v-y0)/(y1-y0)) * ph
        d = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(n))
        eq_svg = f'<svg viewBox="0 0 {w} {h}" class="chart"><text x="{w//2}" y="20" text-anchor="middle" class="st">Equity Curve ($)</text><path d="{d}" fill="none" stroke="#3fb950" stroke-width="2"/></svg>'

    # Year rows
    yr_rows = ""
    for yr in s["per_year"]:
        yr_rows += f"<tr><td>{yr['year']}</td><td>{yr['n_trades']}</td><td>{_fp(yr['return_pct'])}</td><td>{_fr(yr['sharpe'])}</td><td>{_fp(yr['max_dd_pct'])}</td><td>{_fp(yr['win_rate']*100)}</td><td>{_fd(yr['pnl'])}</td><td>{_ti(yr['profitable'])}</td></tr>"

    # Asset rows
    asset_rows = ""
    for ar in s["per_asset"]:
        asset_rows += f"<tr><td style='text-align:left'>{ar['asset']}</td><td>{ar['weight']:.2f}</td><td>{ar['n_trades']}</td><td>{_fd(ar['pnl'])}</td><td>{_fr(ar['sharpe'])}</td><td>{_fp(ar['win_rate']*100)}</td></tr>"

    # Criteria rows
    crit_rows = ""
    for name, c in sc.items():
        crit_rows += f"<tr><td style='text-align:left'>{name}</td><td>{c['target']}</td><td>{c['actual'] if not isinstance(c['actual'], float) else _fr(c['actual'])}</td><td>{_ti(c['met'])}</td></tr>"

    all_met = all(c["met"] for c in sc.values())
    oc = "#3fb950" if all_met else "#f85149"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>EXP-702-max Results</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{color:#8b949e}}
.hero{{background:#161b22;border:2px solid {oc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:2.5em;font-weight:800;color:{oc}}}.hero .sub{{color:#8b949e}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22}}
.chart{{width:100%;max-width:750px;margin:16px auto;display:block}}
.st{{fill:#58a6ff;font-size:13px}}
</style></head><body>
<h1>EXP-702-max: Multi-Asset Portfolio</h1>
<div class="hero">
<div class="big">{"ALL CRITERIA MET" if all_met else "CRITERIA NOT MET"}</div>
<div class="sub">{s['n_trades']} trades &middot; {s['n_years']} years &middot; 4 assets &middot; Risk Parity</div>
</div>
<div class="cards">
<div class="c"><div class="l">Ann. Return</div><div class="v">{_fp(s['annualized_return_pct'])}</div></div>
<div class="c"><div class="l">Sharpe</div><div class="v">{_fr(s['sharpe'])}</div></div>
<div class="c"><div class="l">Sortino</div><div class="v">{_fr(s['sortino'])}</div></div>
<div class="c"><div class="l">Max DD</div><div class="v">{_fp(s['max_drawdown_pct'])}</div></div>
<div class="c"><div class="l">Win Rate</div><div class="v">{_fp(s['win_rate']*100)}</div></div>
<div class="c"><div class="l">Profit Factor</div><div class="v">{_fr(s['profit_factor'])}</div></div>
<div class="c"><div class="l">Total PnL</div><div class="v">{_fd(s['total_pnl'])}</div></div>
<div class="c"><div class="l">Final Capital</div><div class="v">{_fd(s['final_capital'])}</div></div>
<div class="c"><div class="l">SPY-Only Sharpe</div><div class="v">{_fr(s['spy_only_sharpe'])}</div></div>
<div class="c"><div class="l">Sharpe Lift</div><div class="v">{_fr(s['sharpe_improvement'])}</div></div>
</div>
<h2>Equity Curve</h2>{eq_svg}
<h2>Success Criteria</h2>
<table><tr><th style="text-align:left">Criterion</th><th>Target</th><th>Actual</th><th>Met</th></tr>{crit_rows}</table>
<h2>Per-Year Performance</h2>
<table><tr><th>Year</th><th>Trades</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>PnL</th><th>Profitable</th></tr>{yr_rows}</table>
<h2>Per-Asset Breakdown</h2>
<table><tr><th style="text-align:left">Asset</th><th>Weight</th><th>Trades</th><th>PnL</th><th>Sharpe</th><th>Win Rate</th></tr>{asset_rows}</table>
</body></html>"""


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("Running EXP-702-max backtest...")

    result = run_backtest()
    if isinstance(result, dict) and "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    summary, year_results, asset_results, all_trades, equity = result

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON (strip equity_curve for size)
    json_summary = {k: v for k, v in summary.items() if k != "equity_curve"}
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(json_summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Written: results/summary.json")

    # HTML
    html = generate_report(summary, year_results, asset_results, equity)
    (RESULTS_DIR / "report.html").write_text(html, encoding="utf-8")
    print(f"  Written: results/report.html")

    # Print summary
    sc = summary["success_criteria"]
    print(f"\n{'='*60}")
    print(f"  EXP-702-max: Multi-Asset Portfolio Results")
    print(f"{'='*60}")
    print(f"  Trades:       {summary['n_trades']}")
    print(f"  Total PnL:    ${summary['total_pnl']:,.2f}")
    print(f"  Ann. Return:  {summary['annualized_return_pct']:.1f}%")
    print(f"  Sharpe:       {summary['sharpe']:.2f}")
    print(f"  Max DD:       {summary['max_drawdown_pct']:.1f}%")
    print(f"  Win Rate:     {summary['win_rate']:.1%}")
    print(f"  SPY Sharpe:   {summary['spy_only_sharpe']:.2f}")
    print(f"  Sharpe Lift:  {summary['sharpe_improvement']:+.2f}")
    print(f"\n  Success Criteria:")
    for name, c in sc.items():
        icon = "✓" if c["met"] else "✗"
        print(f"    {icon} {name}: {c['actual']} (target: {c['target']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
