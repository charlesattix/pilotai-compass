"""EXP-2500 — TRUE Net Backtest with Cost-Aware Parameters.

EXP-2440 modeled the impact of cost-aware levers (wider spreads, 21d
cadence, single-leg) as sensitivity projections. EXP-2500 replaces
those projections with a REAL walk-forward backtest:

  1. Re-run EXP-1220 with 21-day minimum spacing (vs baseline 10d)
     and a wider-strike selection (otm_pct=0.93 ≈ 20-25 delta, vs
     baseline 0.95 ≈ 30 delta)
  2. Apply per-trade dollar costs to each trade using the EXP-2420
     cost model (commission + bid-ask + slippage); subtract from pnl
  3. Build the 7-stream cube with the cost-aware EXP-1220 and the
     existing 6 streams; apply per-stream annual drag to the other
     streams at their equal_risk weights (EXP-2420 measurement)
  4. Run Ledoit-Wolf risk-parity walk-forward (EXP-2450 methodology,
     20 folds) on the NET cube
  5. Compare NET Sharpe / CAGR / DD against:
        - EXP-2450 gross Ledoit-Wolf (6.87, 101.8%, 4.2% DD)
        - EXP-2420 published net (4.49, 124.0%, unknown DD)

KEY QUESTION
============
Does net Sharpe cross 6.0 with cost-aware parameters on a TRUE
walk-forward (not a sensitivity projection)?

HONEST NULL HYPOTHESIS
======================
The cost-aware parameters REDUCE per-trade alpha (wider strikes
capture less premium, longer holds degrade theta decay). The lever
projections in EXP-2440 assumed gross Sharpe preservation — here
we measure what actually happens.

Rule Zero: real IronVault SPY chains, real commission + slippage
model. No synthetic.

OUTPUT
  compass/reports/exp2500_true_net_backtest.json
  compass/reports/exp2500_true_net_backtest.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2500_true_net_backtest.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2500_true_net_backtest.html"

CACHE_DIR = ROOT / "compass" / "cache"
CACHE_EXP1220_COST_AWARE = CACHE_DIR / "exp2500_exp1220_21d_93otm.pkl"

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000
TRADING_DAYS = 252

# EXP-2420 per-stream annual drag (dollars on $100K at equal_risk weights)
EXP2420_DRAG_USD = {
    "exp1220":  979.17,
    "v5_hedge": 1064.00,
    "gld_cal":  4583.00,
    "slv_cal":  6976.00,
    "cross_vol": 5651.00,
    "xlf_cs":   1911.61,
    "xli_cs":   1040.45,
}
EXP2420_BASELINE_WEIGHTS = {
    "exp1220": 0.316, "v5_hedge": 0.023, "gld_cal": 0.024, "slv_cal": 0.012,
    "cross_vol": 0.187, "xlf_cs": 0.245, "xli_cs": 0.192,
}

# EXP-2420 cost model per-trade on baseline EXP-1220
# (3 contracts, 2 legs, $0.65 commission, bid-ask from IronVault p25, slippage ADV-based)
BASELINE_EXP1220_PER_TRADE_USD = 28.80    # total per-trade cost
BASELINE_EXP1220_CONTRACTS = 3.0
BASELINE_EXP1220_PER_CONTRACT_USD = BASELINE_EXP1220_PER_TRADE_USD / BASELINE_EXP1220_CONTRACTS  # 9.60


# ═══════════════════════════════════════════════════════════════════════════
# 1. Build cost-aware EXP-1220 trade tape
# ═══════════════════════════════════════════════════════════════════════════

def run_cost_aware_exp1220() -> List[Dict]:
    """Fork of run_exp1220_trades with:
       - min_spacing_days = 21 (biweekly → 3-weekly)
       - otm_pct = 0.93 (5% OTM → 7% OTM ≈ ~20-25 delta on 30 DTE)
    """
    if CACHE_EXP1220_COST_AWARE.exists():
        print(f"[cache] cost-aware exp1220 from {CACHE_EXP1220_COST_AWARE.name}")
        with open(CACHE_EXP1220_COST_AWARE, "rb") as fh:
            return pickle.load(fh)

    print("[run] building cost-aware exp1220 tape (real IronVault)...")
    import yfinance as yf
    from shared.iron_vault import IronVault
    from compass.exp1220_standalone import (
        _find_exps, _exp_dt, _next_td, _sell_put_spread, _walk_spread,
    )

    hd = IronVault.instance()
    spy_df = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)

    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "2020-03-01", "2025-12-31", monthly=False)
    trades: List[Dict] = []
    last = None
    MIN_SPACING_DAYS = 21     # cost-aware cadence
    OTM_PCT = 0.93            # cost-aware wider strike

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < MIN_SPACING_DAYS:
            continue
        try:
            price = float(spy_close.loc[es])
            v = float(vix.loc[es])
        except Exception:
            continue
        if np.isnan(price) or np.isnan(v) or v > 40:
            continue

        spread = _sell_put_spread(hd, exp, es, price,
                                   otm_pct=OTM_PCT, width=5.0)
        if spread is None:
            continue
        cts = max(1, min(4, int(CAPITAL * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index,
        )
        gross_pnl = (spread["credit"] - ev) * 100 * cts

        # PER-TRADE COST (EXP-2420 cost model)
        # commission: $0.65 × 2 legs × 2 fills (open+close) × contracts
        commission = 0.65 * 2 * 2 * cts
        # bid-ask: half the option bid-ask spread, applied to both legs,
        # both fills. Proxy: $1.70 per leg per contract (from EXP-2420
        # per-trade bid_ask $10.25 / 3 cts / 2 fills)
        bid_ask = 1.70 * 2 * 2 * cts
        # slippage: ADV-based, ~$3.58/contract (from EXP-2420
        # slippage_per_trade $10.75 / 3 cts)
        slippage = 3.58 * cts
        cost = commission + bid_ask + slippage
        net_pnl = gross_pnl - cost

        trades.append({
            "entry_date": es,
            "exit_date": ed,
            "gross_pnl": round(gross_pnl, 2),
            "cost_usd": round(cost, 2),
            "pnl": round(net_pnl, 2),
            "exit_reason": er,
            "credit": spread["credit"],
            "vix": round(v, 1),
            "hold_days": hold,
            "contracts": cts,
            "width": spread["width"],
            "max_loss": spread["max_loss"],
        })
        last = entry_dt

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_EXP1220_COST_AWARE, "wb") as fh:
        pickle.dump(trades, fh)
    return trades


def trade_tape_summary(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0}
    gross = np.array([t["gross_pnl"] for t in trades])
    cost = np.array([t["cost_usd"] for t in trades])
    net = np.array([t["pnl"] for t in trades])
    wins_gross = int((gross > 0).sum())
    wins_net = int((net > 0).sum())
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"])
    ex = pd.to_datetime(df["exit_date"])
    yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
    tpy = len(trades) / yrs
    return {
        "label": label,
        "n_trades": len(trades),
        "trades_per_year": round(tpy, 2),
        "gross_total_pnl": round(float(gross.sum()), 2),
        "net_total_pnl": round(float(net.sum()), 2),
        "total_cost_usd": round(float(cost.sum()), 2),
        "avg_cost_per_trade": round(float(cost.mean()), 2),
        "gross_avg_pnl": round(float(gross.mean()), 2),
        "net_avg_pnl": round(float(net.mean()), 2),
        "gross_win_rate": round(wins_gross / len(trades), 4),
        "net_win_rate": round(wins_net / len(trades), 4),
        "avg_hold_days": round(float(df["hold_days"].mean()), 1),
    }


def trades_to_daily(trades: List[Dict], index: pd.DatetimeIndex,
                     pnl_field: str = "pnl") -> pd.Series:
    """Sparse exit-date conversion (matches EXP-2450 methodology)."""
    if not trades:
        return pd.Series(0.0, index=index)
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")[pnl_field].sum() / CAPITAL
    return daily.reindex(index, fill_value=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Apply per-stream drag to the other 6 streams
# ═══════════════════════════════════════════════════════════════════════════

def apply_stream_drag(series: pd.Series, stream_name: str,
                       equal_risk_weight: float) -> pd.Series:
    """Subtract the per-stream daily drag from the stream's return series.

    EXP-2420 measured the annual dollar drag at baseline equal_risk
    weights. We convert to a daily return deduction AT THE BASELINE
    WEIGHT — which means the drag is proportional to the stream's
    weight in the portfolio. The walk-forward may re-weight, but the
    drag model here is a conservative baseline (real drag scales
    with actual contract count, not weight).
    """
    drag_usd = EXP2420_DRAG_USD.get(stream_name, 0.0)
    w = EXP2420_BASELINE_WEIGHTS.get(stream_name, 1.0)
    # Drag per day as a fraction of capital, normalized to weight=1 so
    # the portfolio cost scales linearly with the stream's weight.
    daily_drag_per_unit_weight = drag_usd / TRADING_DAYS / CAPITAL / max(w, 1e-6)
    return series - daily_drag_per_unit_weight * w


# ═══════════════════════════════════════════════════════════════════════════
# 3. Build the NET 7-stream cube
# ═══════════════════════════════════════════════════════════════════════════

def build_net_cube() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Returns (gross_cube, net_cube, per_stream_info).

    gross_cube  = reference cube matching EXP-2450 (for comparison)
    net_cube    = cube with cost-aware exp1220 + drag-adjusted siblings
    """
    from compass.exp2080_corr_regime import load_streams as load_5stream
    from compass.exp2160_high_capacity_alts import (
        run_put_credit_spreads,
    )
    import sqlite3
    from shared.iron_vault import IronVault

    print("\n[cube] loading 5-stream base...")
    base = load_5stream()
    print(f"       {base.shape}")

    # Use EXP-2450's sparse XLF/XLI convention (exit-date, not smeared)
    print("[cube] building sparse XLF/XLI streams...")
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    xlf_trades = run_put_credit_spreads(con, "XLF")
    xli_trades = run_put_credit_spreads(con, "XLI")
    con.close()

    def spread_to_sparse(trades, idx):
        s = pd.Series(0.0, index=idx)
        for t in trades:
            try:
                d = pd.Timestamp(t.expiration)
                if d in s.index:
                    s.loc[d] += float(t.pnl_pct_capital)
            except Exception:
                pass
        return s

    xlf_gross = spread_to_sparse(xlf_trades, base.index)
    xli_gross = spread_to_sparse(xli_trades, base.index)

    # Cost-aware EXP-1220
    print("[cube] building cost-aware EXP-1220 (21d cadence, 0.93 OTM)...")
    ca_trades = run_cost_aware_exp1220()
    ca_summary_gross = trade_tape_summary(ca_trades, "cost_aware_gross")
    ca_summary_net = trade_tape_summary(ca_trades, "cost_aware_net")
    print(f"       n_trades: {ca_summary_gross['n_trades']}")
    print(f"       trades/year: {ca_summary_gross['trades_per_year']:.1f}")
    print(f"       gross avg_pnl: ${ca_summary_gross['gross_avg_pnl']}")
    print(f"       cost/trade: ${ca_summary_gross['avg_cost_per_trade']}")
    print(f"       net avg_pnl: ${ca_summary_net['net_avg_pnl']}")
    print(f"       gross WR: {ca_summary_gross['gross_win_rate']*100:.1f}%  "
          f"net WR: {ca_summary_net['net_win_rate']*100:.1f}%")

    # Daily series from trades
    idx = base.index
    ca_gross_daily = trades_to_daily(ca_trades, idx, pnl_field="gross_pnl")
    ca_net_daily = trades_to_daily(ca_trades, idx, pnl_field="pnl")

    # ── GROSS cube (reference — reproduce EXP-2450 Ledoit-Wolf gross)
    gross_cube = base.copy()
    gross_cube["xlf_cs"] = xlf_gross
    gross_cube["xli_cs"] = xli_gross
    gross_cube = gross_cube[["exp1220", "v5_hedge", "gld_cal", "slv_cal",
                              "cross_vol", "xlf_cs", "xli_cs"]].fillna(0.0)

    # ── NET cube: replace exp1220 with cost-aware net, apply drag to others
    net_cube = gross_cube.copy()
    net_cube["exp1220"] = ca_net_daily.reindex(idx, fill_value=0.0)

    # Apply per-stream drag to the other 6 streams (dollar drag / 252 / capital
    # converted to daily return reduction, scaled by equal_risk baseline weight)
    for col in ["v5_hedge", "gld_cal", "slv_cal", "cross_vol", "xlf_cs", "xli_cs"]:
        net_cube[col] = apply_stream_drag(net_cube[col], col,
                                           EXP2420_BASELINE_WEIGHTS[col])

    info = {
        "cost_aware_exp1220": {
            "gross_summary": ca_summary_gross,
            "net_summary": ca_summary_net,
            "parameters": {
                "min_spacing_days": 21,
                "otm_pct": 0.93,
                "width": 5.0,
                "risk_pct": 0.03,
                "vix_block": 40.0,
            },
            "cost_model_per_contract": {
                "commission_4fills": 0.65 * 2 * 2,
                "bid_ask_per_leg": 1.70 * 2 * 2,
                "slippage_adv": 3.58,
                "total_per_contract": 0.65 * 2 * 2 + 1.70 * 2 * 2 + 3.58,
            },
        },
        "stream_drag_usd": EXP2420_DRAG_USD,
    }

    return gross_cube, net_cube, info


# ═══════════════════════════════════════════════════════════════════════════
# 4. Walk-forward — reuse EXP-2400 / EXP-2450 infrastructure
# ═══════════════════════════════════════════════════════════════════════════

def run_ledoit_walk_forward(cube: pd.DataFrame, label: str) -> Tuple[pd.Series, List[Dict]]:
    """Invoke EXP-2400's walk_forward_combined with Ledoit-Wolf, no circuit."""
    from compass.exp2400_combined_best_of import walk_forward_combined
    folds, pooled, _lev = walk_forward_combined(
        cube, use_circuit=False, use_ledoit=True,
    )
    return pooled, folds


def pooled_metrics(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"label": label, "n": n}
    mu = float(daily.mean())
    sd = float(daily.std(ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1 + daily).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "label": label,
        "n": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(dd * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2500 — TRUE Net Backtest with Cost-Aware Parameters")
    print("=" * 72)

    gross_cube, net_cube, info = build_net_cube()
    print(f"\n[cubes] built 7-stream cubes, shape {gross_cube.shape}")

    # Sanity: per-stream standalone metrics on both cubes
    print("\n[per-stream] standalone pooled metrics (full sample):")
    per_stream = {}
    for col in gross_cube.columns:
        g_m = pooled_metrics(gross_cube[col], f"{col}_gross")
        n_m = pooled_metrics(net_cube[col], f"{col}_net")
        per_stream[col] = {"gross": g_m, "net": n_m}
        print(f"  {col:10s}  gross SR {g_m['sharpe']:6.2f} "
              f"CAGR {g_m['cagr_pct']:+6.1f}%  |  "
              f"net SR {n_m['sharpe']:6.2f} CAGR {n_m['cagr_pct']:+6.1f}%")

    # Run walk-forward on both cubes
    print("\n[walk-forward] gross cube (Ledoit-Wolf, 20 folds)...")
    gross_pooled, gross_folds = run_ledoit_walk_forward(gross_cube, "gross")
    gross_m = pooled_metrics(gross_pooled, "gross_pooled")
    print(f"       pooled: CAGR {gross_m['cagr_pct']:+7.1f}%  "
          f"SR {gross_m['sharpe']:5.2f}  "
          f"DD {gross_m['max_dd_pct']:5.1f}%  "
          f"n_folds {len(gross_folds)}")

    print("\n[walk-forward] net cube (Ledoit-Wolf, 20 folds, NET exp1220+drag)...")
    net_pooled, net_folds = run_ledoit_walk_forward(net_cube, "net")
    net_m = pooled_metrics(net_pooled, "net_pooled")
    print(f"       pooled: CAGR {net_m['cagr_pct']:+7.1f}%  "
          f"SR {net_m['sharpe']:5.2f}  "
          f"DD {net_m['max_dd_pct']:5.1f}%  "
          f"n_folds {len(net_folds)}")

    # Reference numbers for comparison
    exp2450_ledoit_gross = {"sharpe": 6.87, "cagr_pct": 101.8, "max_dd_pct": 4.2}
    exp2420_published_net = {"sharpe": 4.49, "cagr_pct": 124.0}

    # Yearly breakdown on net pooled
    print("\n[yearly] net pooled metrics by year:")
    yearly_net = {}
    for yr in sorted({d.year for d in net_pooled.index}):
        sub = net_pooled[net_pooled.index.year == yr]
        if len(sub) < 20:
            continue
        m = pooled_metrics(sub, f"year_{yr}")
        yearly_net[int(yr)] = m
        print(f"       {yr}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"SR {m['sharpe']:5.2f}  DD {m['max_dd_pct']:5.1f}%")

    # ── Verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  EXP-2450 gross ledoit_only (baseline):   SR 6.87  CAGR 101.8%  DD 4.2%")
    print(f"  EXP-2420 published net (static params):  SR 4.49  CAGR 124.0%  DD n/a")
    print(f"  EXP-2500 NET gross (sanity check):       SR {gross_m['sharpe']}  "
          f"CAGR {gross_m['cagr_pct']}%  DD {gross_m['max_dd_pct']}%")
    print(f"  EXP-2500 NET net (cost-aware):           SR {net_m['sharpe']}  "
          f"CAGR {net_m['cagr_pct']}%  DD {net_m['max_dd_pct']}%")
    print()
    cross_6 = net_m["sharpe"] > 6.0
    print(f"  Does net Sharpe cross 6.0? {'YES ✓' if cross_6 else 'NO ✗'}")
    print(f"  Delta from EXP-2450 gross baseline:   "
          f"{net_m['sharpe'] - 6.87:+.2f}")
    print(f"  Delta from EXP-2420 published net:    "
          f"{net_m['sharpe'] - 4.49:+.2f}")

    payload = {
        "experiment": "EXP-2500",
        "title": "TRUE Net Backtest with Cost-Aware Parameters",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "cost_aware_exp1220": "real IronVault SPY chains, min_spacing=21d, otm_pct=0.93",
            "per_trade_cost_model": "EXP-2420 commission + bid-ask + slippage, applied per-contract",
            "other_streams_drag": "EXP-2420 per-stream annual drag at equal_risk weights",
            "walk_forward": "compass.exp2400_combined_best_of.walk_forward_combined (LW, no circuit)",
            "cubes": "sparse exit-date convention (matches EXP-2450)",
        },
        "config": {
            "min_spacing_days": 21,
            "otm_pct": 0.93,
            "width": 5.0,
            "train_days": 252,
            "test_days": 63,
            "target_vol_annual": 0.15,
        },
        "cost_aware_exp1220_info": info["cost_aware_exp1220"],
        "stream_drag_usd": EXP2420_DRAG_USD,
        "per_stream_gross_vs_net": per_stream,
        "walk_forward": {
            "gross": {"pooled": gross_m, "n_folds": len(gross_folds)},
            "net": {"pooled": net_m, "n_folds": len(net_folds),
                     "yearly": yearly_net},
        },
        "references": {
            "exp2450_gross_ledoit_only": exp2450_ledoit_gross,
            "exp2420_published_net": exp2420_published_net,
        },
        "verdict": {
            "net_sharpe": net_m["sharpe"],
            "net_cagr_pct": net_m["cagr_pct"],
            "net_max_dd_pct": net_m["max_dd_pct"],
            "crosses_6_0": cross_6,
            "delta_vs_gross_ledoit": round(net_m["sharpe"] - 6.87, 3),
            "delta_vs_published_net": round(net_m["sharpe"] - 4.49, 3),
        },
        "honest_caveats": [
            "Per-trade cost uses EXP-2420's per-contract rates, applied to the NEW trade contract counts. Real broker fills may have different bid-ask at the time of trade; IronVault p25 spread is an average.",
            "The cost-aware EXP-1220 variant (21d cadence, 0.93 otm_pct) may have different chain liquidity and slippage than the EXP-2420 baseline measurement.",
            "Other 6 streams' drag is applied at equal_risk baseline weights. The walk-forward optimizer reweights per fold, so actual stream weights differ from baseline — drag is approximately linear in weight but not exactly.",
            "Longer holds (21d) and wider strikes (0.93) may reduce per-trade alpha. The backtest measures the actual impact.",
            "Gross cube sanity check should reproduce EXP-2450 ledoit_only (6.87 / 101.8%). If it does not, there is a cube-build mismatch.",
        ],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    html = build_html(payload)
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    wf = p["walk_forward"]
    net = wf["net"]["pooled"]
    gross = wf["gross"]["pooled"]
    ref_gross = p["references"]["exp2450_gross_ledoit_only"]
    ref_net = p["references"]["exp2420_published_net"]
    v = p["verdict"]
    ca = p["cost_aware_exp1220_info"]
    ca_g = ca["gross_summary"]
    ca_n = ca["net_summary"]

    yr_rows = ""
    for yr, m in wf["net"]["yearly"].items():
        color = "#16a34a" if m["cagr_pct"] > 0 else "#dc2626"
        yr_rows += (
            f"<tr><td>{yr}</td>"
            f"<td style='color:{color}'>{m['cagr_pct']:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td></tr>"
        )

    ps_rows = ""
    for col, both in p["per_stream_gross_vs_net"].items():
        g, n = both["gross"], both["net"]
        ps_rows += (
            f"<tr><td><strong>{col}</strong></td>"
            f"<td>{g.get('sharpe', 0):.2f}</td>"
            f"<td>{g.get('cagr_pct', 0):+.1f}%</td>"
            f"<td>{n.get('sharpe', 0):.2f}</td>"
            f"<td>{n.get('cagr_pct', 0):+.1f}%</td>"
            f"<td>{n.get('sharpe', 0) - g.get('sharpe', 0):+.2f}</td></tr>"
        )

    cross_color = "#16a34a" if v["crosses_6_0"] else "#dc2626"
    cross_text = "YES ✓" if v["crosses_6_0"] else "NO ✗"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2500 — True Net Backtest</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.verdict {{ background:#fff;border:2px solid {cross_color};border-radius:10px;padding:18px;margin:16px 0; }}
.verdict h3 {{ margin-top:0;color:{cross_color}; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2500 — TRUE Net Backtest with Cost-Aware Parameters</h1>
<p style="color:#64748b">Actual per-trade cost + cost-aware EXP-1220 parameters
(21d cadence, 0.93 OTM) · 7-stream Ledoit-Wolf walk-forward ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — real data, real per-trade costs:</strong><br>
Cost-aware EXP-1220: compass.exp1220_standalone fork with
<code>min_spacing_days=21</code> and <code>otm_pct=0.93</code> on real
IronVault SPY chains<br>
Per-trade cost: EXP-2420 model ($0.65/contract commission × 4 fills +
$1.70 bid-ask per leg per contract × 4 + $3.58/contract slippage)<br>
Other streams: sparse XLF/XLI (exit-date), v5/gld/slv/cross_vol from
compass.exp2080_corr_regime cache<br>
Other-stream drag: EXP-2420 measured annual drag at equal_risk weights,
applied as daily deduction<br>
Walk-forward: compass.exp2400_combined_best_of.walk_forward_combined
(Ledoit-Wolf + risk-parity + 15% vol target, no circuit breaker),
20 folds 2020-2025
</div>

<div class="verdict">
<h3>Key question: does net Sharpe cross 6.0?</h3>
<strong>{cross_text}</strong><br>
Net Sharpe: <strong>{net['sharpe']:.2f}</strong> ·
Net CAGR: <strong>{net['cagr_pct']:+.1f}%</strong> ·
Net DD: <strong>{net['max_dd_pct']:.1f}%</strong><br>
Δ vs EXP-2450 gross LW (6.87): <strong>{v['delta_vs_gross_ledoit']:+.2f}</strong><br>
Δ vs EXP-2420 published net (4.49): <strong>{v['delta_vs_published_net']:+.2f}</strong>
</div>

<h2>1. Cost-aware EXP-1220 trade tape</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>N trades</td><td>{ca_g['n_trades']}</td></tr>
<tr><td>Trades/year</td><td>{ca_g['trades_per_year']}</td></tr>
<tr><td>Avg hold days</td><td>{ca_g['avg_hold_days']}</td></tr>
<tr><td>Gross avg P&L / trade</td><td>${ca_g['gross_avg_pnl']}</td></tr>
<tr><td>Avg cost / trade</td><td>${ca_g['avg_cost_per_trade']}</td></tr>
<tr><td>Net avg P&L / trade</td><td>${ca_n['net_avg_pnl']}</td></tr>
<tr><td>Gross total P&L</td><td>${ca_g['gross_total_pnl']:,}</td></tr>
<tr><td>Total cost</td><td>${ca_g['total_cost_usd']:,}</td></tr>
<tr><td>Net total P&L</td><td>${ca_n['net_total_pnl']:,}</td></tr>
<tr><td>Gross win rate</td><td>{ca_g['gross_win_rate']*100:.1f}%</td></tr>
<tr><td>Net win rate</td><td>{ca_n['net_win_rate']*100:.1f}%</td></tr>
</tbody>
</table>

<h2>2. 7-stream walk-forward (Ledoit-Wolf, 20 folds)</h2>
<table>
<thead><tr><th>Cube</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th><th>Calmar</th></tr></thead>
<tbody>
<tr><td><strong>gross (sanity)</strong></td>
<td>{gross['cagr_pct']:.1f}%</td><td>{gross['sharpe']:.2f}</td>
<td>{gross['max_dd_pct']:.1f}%</td><td>{gross['vol_pct']:.1f}%</td>
<td>{gross['calmar']:.1f}</td></tr>
<tr><td><strong>net (cost-aware)</strong></td>
<td>{net['cagr_pct']:.1f}%</td>
<td style='color:{cross_color};font-weight:700'>{net['sharpe']:.2f}</td>
<td>{net['max_dd_pct']:.1f}%</td><td>{net['vol_pct']:.1f}%</td>
<td>{net['calmar']:.1f}</td></tr>
<tr><td>EXP-2450 gross LW reference</td>
<td>{ref_gross['cagr_pct']:.1f}%</td><td>{ref_gross['sharpe']:.2f}</td>
<td>{ref_gross['max_dd_pct']:.1f}%</td><td>—</td><td>—</td></tr>
<tr><td>EXP-2420 published net</td>
<td>{ref_net['cagr_pct']:.1f}%</td><td>{ref_net['sharpe']:.2f}</td>
<td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>

<h2>3. Net yearly breakdown</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{yr_rows}</tbody>
</table>

<h2>4. Per-stream gross vs net (standalone pooled)</h2>
<table>
<thead><tr><th>Stream</th><th>Gross SR</th><th>Gross CAGR</th>
<th>Net SR</th><th>Net CAGR</th><th>ΔSR</th></tr></thead>
<tbody>{ps_rows}</tbody>
</table>

<h2>5. Honest caveats</h2>
<ul>
{''.join(f'<li>{c}</li>' for c in p['honest_caveats'])}
</ul>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2500_true_net_backtest.py · Rule Zero · real data + real costs
</p>
</body></html>"""


if __name__ == "__main__":
    main()
