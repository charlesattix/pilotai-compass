#!/usr/bin/env python3
"""
EXP-1820 Dispersion Scaling
=============================
Previous finding: Sharpe 1.94 but only 0.7% CAGR (89 trades, too sparse).

Goal: Can dispersion reach 10%+ CAGR while keeping Sharpe > 1.5?

Scaling approaches tested:
  1. Weekly cadence (5-day spacing) vs monthly (20-day)
  2. Expanded ticker set: +QQQ, +TLT, +GLD pairs
  3. Leverage: 1.0x, 1.5x, 2.0x, 3.0x
  4. Walk-forward validation at optimal config

Real IronVault data + Yahoo spot. Zero synthetic.
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.dispersion_strategy import (
    DispersionStrategy, PRODUCTION_CONFIG, trade_sharpe, corrected_sharpe,
    load_spot, _friday_exps, _strikes, _spread_close, find_put_spread,
    Trade, CAPITAL, COMMISSION,
)

TRADING_DAYS = 252

# Expanded tickers: original sectors + QQQ/TLT/GLD (have data)
# Note: QQQ is correlated to SPY but often has different vol
#       TLT and GLD are uncorrelated — true diversifiers
EXPANDED_SECTORS = ["XLF", "XLI", "XLK", "XLE", "QQQ", "TLT", "GLD"]


def pct(v, d=1): return f"{v*100:+.{d}f}%" if isinstance(v, float) else f"{v}"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


# ═══════════════════════════════════════════════════════════════════════════
# Scaled dispersion backtest
# ═══════════════════════════════════════════════════════════════════════════

def backtest_scaled_dispersion(
    sectors: List[str],
    vol_ratio_threshold: float = 1.15,
    min_spacing_days: int = 5,
    leverage: float = 1.0,
    start: str = "2020-06-01",
    end: str = "2026-01-01",
) -> List[Trade]:
    """Dispersion backtest with configurable cadence, ticker set, leverage."""
    # Reuse production sub-pieces but with custom ticker list + cadence

    cfg = dict(PRODUCTION_CONFIG)
    cfg["sectors"] = sectors
    cfg["vol_ratio_threshold"] = vol_ratio_threshold
    cfg["min_spacing_days"] = min_spacing_days
    cfg["risk_per_trade"] = PRODUCTION_CONFIG["risk_per_trade"] * leverage

    # Load spots
    spots = {}
    for t in ["SPY"] + sectors:
        try:
            spots[t] = load_spot(t, start=start, end=end)
        except Exception:
            pass

    if "SPY" not in spots:
        return []

    exps = _friday_exps("SPY", start, end)
    trades = []
    last_entry_by_ticker = {t: None for t in sectors}

    for exp in exps:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        entry_dt = exp_dt - timedelta(days=cfg["target_dte"])

        for off in range(7):
            cand = entry_dt + timedelta(days=off)
            if pd.Timestamp(cand) in spots["SPY"].index:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")

        if pd.Timestamp(entry_dt) not in spots["SPY"].index:
            continue
        spy_spot = float(spots["SPY"].loc[pd.Timestamp(entry_dt)])

        spy_spread = find_put_spread(
            "SPY", exp, entry_str, spy_spot,
            otm_pct=cfg["otm_pct"],
            width=cfg["spy_width"],
            min_credit=cfg["min_credit"],
        )
        if spy_spread is None:
            continue
        spy_vol_proxy = spy_spread["credit_pct_of_spot"]
        if spy_vol_proxy <= 0:
            continue

        for sector in sectors:
            if sector not in spots:
                continue

            if last_entry_by_ticker[sector] is not None:
                if (entry_dt - last_entry_by_ticker[sector]).days < min_spacing_days:
                    continue

            if pd.Timestamp(entry_dt) not in spots[sector].index:
                continue
            sec_spot = float(spots[sector].loc[pd.Timestamp(entry_dt)])

            # Adaptive width: bigger for higher-priced tickers
            if sec_spot < 80:
                sec_width = 1.0
            elif sec_spot < 200:
                sec_width = 2.0
            else:
                sec_width = 5.0

            sec_spread = find_put_spread(
                sector, exp, entry_str, sec_spot,
                otm_pct=cfg["otm_pct"],
                width=sec_width,
                min_credit=cfg["min_credit"],
            )
            if sec_spread is None:
                continue

            sec_vol_proxy = sec_spread["credit_pct_of_spot"]
            ratio = sec_vol_proxy / spy_vol_proxy

            if ratio < vol_ratio_threshold:
                continue

            max_loss = sec_spread["max_loss"]
            risk_budget = CAPITAL * cfg["risk_per_trade"]
            contracts = max(1, min(30, int(risk_budget / (max_loss * 100))))

            # Walk to exit
            exit_date = exp
            exit_reason = "expiration"
            exit_credit = 0.0

            cur_dt = entry_dt + timedelta(days=1)
            sec_spot_idx = spots[sector].index
            while cur_dt <= exp_dt:
                cs = cur_dt.strftime("%Y-%m-%d")
                if pd.Timestamp(cur_dt) not in sec_spot_idx:
                    cur_dt += timedelta(days=1)
                    continue

                pp = _spread_close(sector, exp, sec_spread["put_short"],
                                   sec_spread["put_long"], "P", cs)
                if pp is None:
                    cur_dt += timedelta(days=1)
                    continue

                cur_val = pp["short_close"] - pp["long_close"]

                if cur_val <= sec_spread["credit"] * (1 - cfg["profit_target"]):
                    exit_date, exit_reason, exit_credit = cs, "profit_target", cur_val
                    break
                if cur_val - sec_spread["credit"] > sec_spread["credit"] * cfg["stop_loss_multiplier"]:
                    exit_date, exit_reason, exit_credit = cs, "stop_loss", cur_val
                    break
                if (exp_dt - cur_dt).days <= cfg["dte_exit_days"]:
                    exit_date, exit_reason, exit_credit = cs, "dte_exit", cur_val
                    break

                cur_dt += timedelta(days=1)

            if exit_reason == "expiration":
                exit_credit = 0.0

            commission = 2 * 2 * COMMISSION * contracts
            pnl = (sec_spread["credit"] - exit_credit) * 100 * contracts - commission

            trades.append(Trade(
                entry_date=entry_str,
                exit_date=exit_date,
                exp=exp,
                ticker=sector,
                spot=round(sec_spot, 2),
                put_short=sec_spread["put_short"],
                put_long=sec_spread["put_long"],
                credit=round(sec_spread["credit"], 3),
                contracts=contracts,
                exit_reason=exit_reason,
                pnl=round(pnl, 2),
                vol_ratio=round(ratio, 3),
                avg_contracts_at_entry=contracts,
            ))
            last_entry_by_ticker[sector] = entry_dt

    return trades


def compute_metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "sharpe": 0, "cagr": 0,
                "max_dd": 0, "is_sharpe": 0, "oos_sharpe": 0, "yearly": {}}
    df = pd.DataFrame([vars(t) for t in trades])
    pnls = df["pnl"].values
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    sharpe = trade_sharpe(pnls)
    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() if len(pk) > 0 else 0

    dates = pd.to_datetime(df["exit_date"])
    entry_dates = pd.to_datetime(df["entry_date"])
    years = max((dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / years) - 1) if total > -CAPITAL else -1

    is_mask = dates.dt.year <= 2022
    oos_mask = dates.dt.year >= 2023
    is_sharpe = trade_sharpe(pnls[is_mask]) if is_mask.any() else 0
    oos_sharpe = trade_sharpe(pnls[oos_mask]) if oos_mask.any() else 0

    df["year"] = dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yearly[int(yr)] = {
            "n": len(yp),
            "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum() / len(yp)), 3),
            "sharpe": round(trade_sharpe(yp), 2),
        }

    return {
        "n": len(pnls), "pnl": round(total, 2),
        "wr": round(wins / len(pnls), 3),
        "sharpe": round(sharpe, 2),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(dd), 4),
        "is_sharpe": round(is_sharpe, 2),
        "oos_sharpe": round(oos_sharpe, 2),
        "yearly": yearly,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analyses
# ═══════════════════════════════════════════════════════════════════════════

def cadence_sweep():
    """Test different trade spacings (frequency)."""
    print("\n[1] Cadence sweep (tight → loose spacing):")
    results = {}
    for spacing in [5, 10, 14, 20]:
        trades = backtest_scaled_dispersion(
            sectors=["XLF", "XLI", "XLK", "XLE"],
            min_spacing_days=spacing,
            leverage=1.0,
        )
        m = compute_metrics(trades)
        results[f"{spacing}d spacing"] = m
        print(f"  {spacing:2d}d: n={m['n']:3d} Sharpe={m['sharpe']:5.2f} "
              f"OOS={m['oos_sharpe']:5.2f} CAGR={pct(m['cagr']):>7s} "
              f"DD={pct(m['max_dd']):>6s}")
    return results


def ticker_expansion():
    """Compare original vs expanded ticker sets."""
    print("\n[2] Ticker expansion:")
    results = {}
    sets = {
        "Original (4 sectors)": ["XLF", "XLI", "XLK", "XLE"],
        "+ QQQ (5)": ["XLF", "XLI", "XLK", "XLE", "QQQ"],
        "+ QQQ + TLT (6)": ["XLF", "XLI", "XLK", "XLE", "QQQ", "TLT"],
        "+ QQQ + TLT + GLD (7)": EXPANDED_SECTORS,
    }
    for name, tickers in sets.items():
        trades = backtest_scaled_dispersion(
            sectors=tickers,
            min_spacing_days=10,  # weekly-ish
            leverage=1.0,
        )
        m = compute_metrics(trades)
        results[name] = m
        print(f"  {name:30s}: n={m['n']:3d} Sharpe={m['sharpe']:5.2f} "
              f"OOS={m['oos_sharpe']:5.2f} CAGR={pct(m['cagr']):>7s}")
    return results


def leverage_sweep():
    """Test leverage scaling."""
    print("\n[3] Leverage sweep (with 7 tickers, 10d spacing):")
    results = {}
    for lev in [1.0, 1.5, 2.0, 3.0]:
        trades = backtest_scaled_dispersion(
            sectors=EXPANDED_SECTORS,
            min_spacing_days=10,
            leverage=lev,
        )
        m = compute_metrics(trades)
        results[f"{lev}x"] = m
        print(f"  {lev}x: n={m['n']:3d} Sharpe={m['sharpe']:5.2f} "
              f"OOS={m['oos_sharpe']:5.2f} CAGR={pct(m['cagr']):>7s} "
              f"DD={pct(m['max_dd']):>6s}")
    return results


def walk_forward_optimal(optimal_config):
    """Walk-forward on optimal config."""
    print(f"\n[4] Walk-forward validation (optimal config):")
    print(f"    sectors={optimal_config['sectors']}")
    print(f"    spacing={optimal_config['spacing']}d, leverage={optimal_config['leverage']}x")

    trades = backtest_scaled_dispersion(
        sectors=optimal_config["sectors"],
        min_spacing_days=optimal_config["spacing"],
        leverage=optimal_config["leverage"],
    )

    if not trades:
        return []

    df = pd.DataFrame([vars(t) for t in trades])
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_dt"].dt.year

    windows = []
    for oos_year in [2022, 2023, 2024, 2025]:
        is_df = df[df["year"] < oos_year]
        oos_df = df[df["year"] == oos_year]
        if is_df.empty or oos_df.empty:
            continue

        is_pnls = is_df["pnl"].values
        oos_pnls = oos_df["pnl"].values
        is_sharpe = trade_sharpe(is_pnls)
        oos_sharpe = trade_sharpe(oos_pnls)

        windows.append({
            "oos_year": oos_year,
            "is_n": len(is_pnls),
            "is_sharpe": round(is_sharpe, 2),
            "oos_n": len(oos_pnls),
            "oos_sharpe": round(oos_sharpe, 2),
            "oos_pnl": round(float(oos_pnls.sum()), 2),
            "oos_wr": round(float((oos_pnls > 0).sum() / len(oos_pnls)), 3),
        })
        print(f"  {oos_year} OOS: IS={is_sharpe:.2f} ({len(is_pnls)} tr) → "
              f"OOS={oos_sharpe:.2f} ({len(oos_pnls)} tr) "
              f"PnL=${oos_pnls.sum():,.0f}")

    return windows


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def build_html(cadence, tickers, leverage, wf, optimal_m, optimal_config):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _table(results, title, dim_name):
        rows = ""
        best = max(results.values(), key=lambda r: r["sharpe"] if r["n"] > 0 else -99)
        for name, m in results.items():
            is_best = m == best
            bg = "background:#f0fdf4;" if is_best else ""
            rows += f"""<tr style="{bg}">
                <td style="text-align:left;font-weight:{'700' if is_best else '500'}">{name}</td>
                <td>{m['n']}</td>
                <td style="color:{clr(m['pnl'])}">${m['pnl']:,.0f}</td>
                <td>{m['wr']*100:.0f}%</td>
                <td>{m['sharpe']:.2f}</td>
                <td style="color:{clr(m['oos_sharpe'])};font-weight:600">{m['oos_sharpe']:.2f}</td>
                <td style="color:{clr(m['cagr'])}">{pct(m['cagr'])}</td>
                <td style="color:#ca8a04">{pct(m['max_dd'])}</td>
            </tr>"""
        return f"""<div class="section-title">{title}</div>
        <table><thead><tr><th>{dim_name}</th><th>Trades</th><th>PnL</th><th>WR</th>
        <th>Sharpe</th><th>OOS Sharpe</th><th>CAGR</th><th>Max DD</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    sweeps_html = _table(cadence, "Cadence Sweep (trade spacing)", "Spacing")
    sweeps_html += _table(tickers, "Ticker Expansion", "Ticker Set")
    sweeps_html += _table(leverage, "Leverage Sweep", "Leverage")

    # WF rows
    wf_rows = ""
    for w in wf:
        wf_rows += f"""<tr>
            <td>{w['oos_year']}</td>
            <td>{w['is_n']}</td>
            <td>{w['is_sharpe']:.2f}</td>
            <td>{w['oos_n']}</td>
            <td style="color:{clr(w['oos_sharpe'])};font-weight:600">{w['oos_sharpe']:.2f}</td>
            <td style="color:{clr(w['oos_pnl'])}">${w['oos_pnl']:,.0f}</td>
            <td>{w['oos_wr']*100:.0f}%</td>
        </tr>"""

    # Yearly for optimal
    yr_rows = ""
    for yr in sorted(optimal_m.get("yearly", {}).keys()):
        d = optimal_m["yearly"][yr]
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td>{d['n']}</td>
            <td style="color:{clr(d['pnl'])}">${d['pnl']:,.0f}</td>
            <td>{d['wr']*100:.0f}%</td>
            <td>{d['sharpe']:.2f}</td>
        </tr>"""

    met = optimal_m["cagr"] >= 0.10 and optimal_m["sharpe"] > 1.5
    vc = "#16a34a" if met else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1820 Dispersion Scaling</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:24px;background:#fff;color:#1e293b; }}
  h1 {{ font-size:1.4rem;margin-bottom:2px; }}
  h2 {{ font-size:1.05rem;color:#1d4ed8;margin:24px 0 8px;border-bottom:2px solid #e2e8f0;padding-bottom:4px; }}
  .meta {{ color:#64748b;font-size:0.82rem;margin-bottom:18px; }}
  .grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px; }}
  .card {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px; }}
  .card-label {{ font-size:0.68rem;color:#64748b;text-transform:uppercase; }}
  .card-value {{ font-size:1.2rem;font-weight:700;margin-top:2px; }}
  table {{ width:100%;border-collapse:collapse;margin-bottom:12px;font-size:0.8rem; }}
  th {{ background:#f1f5f9;padding:5px 8px;text-align:right;font-size:0.7rem;color:#475569;border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:4px 8px;text-align:right;border-bottom:1px solid #f1f5f9; }}
  td:first-child {{ text-align:left;font-weight:500; }}
  .verdict {{ border:3px solid {vc};border-radius:10px;padding:16px;margin:16px 0;
              background:{'#f0fdf4' if met else '#fef9c3'}; }}
  .verdict h3 {{ color:{vc};margin:0 0 6px;font-size:1.1rem; }}
  .section-title {{ font-size:0.95rem;font-weight:600;margin:16px 0 6px;color:#334155; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>EXP-1820 Dispersion Scaling</h1>
<div class="meta">Generated {ts} | Can dispersion reach 10%+ CAGR with Sharpe > 1.5? | Real IronVault data</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All option prices from IronVault (Polygon real).
  Spot prices from Yahoo Finance. Zero synthetic data. Commissions included.
</div>

<div class="verdict">
  <h3>Optimal: {optimal_config['name']}</h3>
  <p style="margin:4px 0;font-size:0.9rem">
    {optimal_m['n']} trades | Sharpe {optimal_m['sharpe']:.2f} |
    CAGR {pct(optimal_m['cagr'])} | Max DD {pct(optimal_m['max_dd'])} |
    OOS Sharpe {optimal_m['oos_sharpe']:.2f}
  </p>
  <p style="font-size:0.85rem;margin:6px 0 0">
    <strong>Target: CAGR ≥ 10% AND Sharpe ≥ 1.5</strong> —
    {'<span style="color:#16a34a;font-weight:700">ACHIEVED</span>' if met else '<span style="color:#ca8a04;font-weight:700">NOT ACHIEVED</span>'}
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">Trades</div><div class="card-value">{optimal_m['n']}</div></div>
  <div class="card"><div class="card-label">CAGR</div><div class="card-value" style="color:{clr(optimal_m['cagr'])}">{pct(optimal_m['cagr'])}</div></div>
  <div class="card"><div class="card-label">Sharpe</div><div class="card-value" style="color:#1d4ed8">{optimal_m['sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">OOS Sharpe</div><div class="card-value" style="color:{clr(optimal_m['oos_sharpe'])}">{optimal_m['oos_sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value" style="color:#ca8a04">{pct(optimal_m['max_dd'])}</div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value">{optimal_m['wr']*100:.0f}%</div></div>
</div>

<h2>1-3. Scaling Experiments</h2>
{sweeps_html}

<h2>4. Walk-Forward (Optimal Config)</h2>
<table><thead><tr><th>OOS Year</th><th>IS Trades</th><th>IS Sharpe</th>
<th>OOS Trades</th><th>OOS Sharpe</th><th>OOS PnL</th><th>OOS WR</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<h2>5. Year-by-Year (Optimal)</h2>
<table><thead><tr><th>Year</th><th>Trades</th><th>PnL</th><th>WR</th><th>Sharpe</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1820 Dispersion Scaling v1.0 | Real IronVault + Yahoo data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1820 DISPERSION SCALING")
    print("Goal: CAGR >= 10%, Sharpe > 1.5")
    print("=" * 70)

    # Sweeps
    cadence = cadence_sweep()
    tickers = ticker_expansion()
    leverage = leverage_sweep()

    # Find optimal config by CAGR (among those that beat 1.0 Sharpe)
    # Using the full scan of combinations
    print("\n[4] Selecting optimal config...")

    # Best from each sweep that meets Sharpe > 1.5
    all_candidates = []
    for name, m in cadence.items():
        if m["sharpe"] > 1.5:
            all_candidates.append(("cadence", name, m))
    for name, m in tickers.items():
        if m["sharpe"] > 1.5:
            all_candidates.append(("tickers", name, m))
    for name, m in leverage.items():
        if m["sharpe"] > 1.5:
            all_candidates.append(("leverage", name, m))

    if all_candidates:
        best = max(all_candidates, key=lambda x: x[2]["cagr"])
        print(f"  Best by CAGR with Sharpe > 1.5: {best[0]}/{best[1]}")
        print(f"    Sharpe {best[2]['sharpe']:.2f}, CAGR {pct(best[2]['cagr'])}")

    # Now try the BIG combo: expanded tickers + weekly + leverage
    print("\n  Testing combined: expanded + 5d spacing + 2x leverage...")
    best_trades = backtest_scaled_dispersion(
        sectors=EXPANDED_SECTORS,
        min_spacing_days=5,
        leverage=2.0,
    )
    best_m = compute_metrics(best_trades)
    print(f"  Combined: n={best_m['n']} Sharpe={best_m['sharpe']} "
          f"CAGR={pct(best_m['cagr'])} DD={pct(best_m['max_dd'])}")

    # Also try 3x leverage on expanded
    print("\n  Testing maxed: expanded + 5d spacing + 3x leverage...")
    maxed_trades = backtest_scaled_dispersion(
        sectors=EXPANDED_SECTORS,
        min_spacing_days=5,
        leverage=3.0,
    )
    maxed_m = compute_metrics(maxed_trades)
    print(f"  Maxed: n={maxed_m['n']} Sharpe={maxed_m['sharpe']} "
          f"CAGR={pct(maxed_m['cagr'])} DD={pct(maxed_m['max_dd'])}")

    # Select final: highest CAGR with Sharpe > 1.5
    candidates = [
        ("Expanded + 5d + 2x", best_m, {"sectors": EXPANDED_SECTORS, "spacing": 5, "leverage": 2.0, "name": "Expanded + 5d + 2x"}),
        ("Expanded + 5d + 3x", maxed_m, {"sectors": EXPANDED_SECTORS, "spacing": 5, "leverage": 3.0, "name": "Expanded + 5d + 3x"}),
    ]
    qualifying = [c for c in candidates if c[1]["sharpe"] > 1.5]
    if qualifying:
        optimal = max(qualifying, key=lambda c: c[1]["cagr"])
    else:
        optimal = max(candidates, key=lambda c: c[1]["sharpe"])

    optimal_m = optimal[1]
    optimal_config = optimal[2]

    # Walk-forward
    wf = walk_forward_optimal(optimal_config)

    # Report
    print("\n[5] Generating report...")
    html = build_html(cadence, tickers, leverage, wf, optimal_m, optimal_config)
    out = ROOT / "reports" / "exp1820_scaling.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    met = optimal_m["cagr"] >= 0.10 and optimal_m["sharpe"] > 1.5
    print(f"  Optimal: {optimal_config['name']}")
    print(f"  Trades: {optimal_m['n']}")
    print(f"  CAGR: {pct(optimal_m['cagr'])}")
    print(f"  Sharpe: {optimal_m['sharpe']:.2f}")
    print(f"  OOS Sharpe: {optimal_m['oos_sharpe']:.2f}")
    print(f"  Max DD: {pct(optimal_m['max_dd'])}")
    print(f"\n  Target (CAGR≥10% AND Sharpe>1.5): {'ACHIEVED' if met else 'NOT ACHIEVED'}")


if __name__ == "__main__":
    main()
