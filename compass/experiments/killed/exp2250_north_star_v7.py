"""EXP-2250 — 8/9-Stream North Star Portfolio v7.

Extends EXP-2200's 7-stream portfolio (sparse Sharpe 5.96 at equal_risk_15%)
with two additional streams:

  8. QQQ put-credit-spread (EXP-2240 discovery: per-trade Sharpe 2.26,
     win rate 90%, correlation to EXP-1220 0.11 — genuinely uncorrelated).
  9. EEM VRP (EXP-1660 survivor: per-trade Sharpe ~2.03 on variance
     swap payoffs, vol-targeted to 5%/yr).

Hypothesis: adding one near-uncorrelated stream should push the sparse
portfolio Sharpe from 5.96 over 6.0. If QQQ alone doesn't do it, EEM
adds a genuinely different underlying (emerging markets) and strategy
(VRP vs credit spreads).

Method (identical to EXP-2200 for apples-to-apples comparison):
  - Reuse the seven-stream cache from EXP-2200
  - Smear trade-tape sleeves uniformly across entry→exit
  - Report BOTH smeared (upper bound) and sparse (honest) daily Sharpe
  - Equal-risk (inverse-vol) weights
  - Test vol targets: none, 12%, 15%
  - Walk-forward 2020-2025 with 252-day warmup trim

Rule Zero: every input traces to real IronVault + Yahoo + FRED.

Output
------
  compass/reports/exp2250_north_star_v7.json
  compass/reports/exp2250_north_star_v7.html
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics

REPORT_JSON = ROOT / "compass" / "reports" / "exp2250_north_star_v7.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2250_north_star_v7.html"

CACHE_DIR = ROOT / "compass" / "cache"
CACHE_V3 = CACHE_DIR / "exp1860_streams.pkl"
CACHE_VOL_ARB = CACHE_DIR / "exp2020_vol_arb_trades.pkl"
CACHE_EXP1220 = CACHE_DIR / "exp2150_trades_biweekly.pkl"
CACHE_XLF = CACHE_DIR / "exp2200_xlf_trades.pkl"
CACHE_XLI = CACHE_DIR / "exp2200_xli_trades.pkl"
CACHE_QQQ = CACHE_DIR / "exp2250_qqq_trades.pkl"
CACHE_EEM_VRP = CACHE_DIR / "exp2250_eem_vrp.pkl"

START = "2020-01-01"
END = "2025-12-31"
WARMUP = 252
CAPITAL = 100_000
EEM_VRP_TARGET_VOL = 0.05  # 5%/yr for vol-targeting EEM VRP stream


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load QQQ credit spread trades (EXP-2240 runner)
# ═══════════════════════════════════════════════════════════════════════════

def load_qqq_trades() -> List[Dict]:
    if CACHE_QQQ.exists():
        print(f"[cache] QQQ CS trades from {CACHE_QQQ.name}")
        with open(CACHE_QQQ, "rb") as fh:
            return pickle.load(fh)

    print("[run] EXP-2240 QQQ credit spreads (real IronVault)...")
    import yfinance as yf
    from shared.iron_vault import IronVault
    from compass.exp2240_qqq_iwm_credit_spreads import run_credit_spread_trades

    hd = IronVault.instance()
    qqq = yf.download("QQQ", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    qqq.index = pd.to_datetime(qqq.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)

    trades = run_credit_spread_trades(hd, "QQQ", qqq, vix, width=5.0)
    print(f"       QQQ: {len(trades)} trades")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_QQQ, "wb") as fh:
        pickle.dump(trades, fh)
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# 2. Load EEM VRP stream (EXP-1660 walk-forward)
# ═══════════════════════════════════════════════════════════════════════════

def load_eem_vrp_stream() -> pd.Series:
    if CACHE_EEM_VRP.exists():
        print(f"[cache] EEM VRP stream from {CACHE_EEM_VRP.name}")
        with open(CACHE_EEM_VRP, "rb") as fh:
            return pickle.load(fh)

    print("[run] EXP-1660 EEM VRP walk-forward (real Yahoo + FRED)...")
    from compass.exp1660_vrp_deepening import load_pair, compute_signals, walk_forward

    try:
        df = load_pair("EEM")
        sig = compute_signals(df)
        bt = walk_forward("EEM", sig)
    except Exception as e:
        print(f"       EEM VRP failed: {e}")
        return pd.Series(dtype=float, name="eem_vrp")

    s = bt.daily_pnl.copy()
    s.index = pd.DatetimeIndex(s.index)

    # Vol-target to 5%/yr (same as EXP-1850 convention)
    realized = s.std() * math.sqrt(252)
    if realized > 1e-9:
        s = s * (EEM_VRP_TARGET_VOL / realized)
    s.name = "eem_vrp"
    print(f"       EEM VRP: {len(s)} days, scaled to {EEM_VRP_TARGET_VOL*100:.0f}%/yr vol")

    with open(CACHE_EEM_VRP, "wb") as fh:
        pickle.dump(s, fh)
    return s


# ═══════════════════════════════════════════════════════════════════════════
# 3. Reusable helpers (copied from EXP-2200 for self-containment)
# ═══════════════════════════════════════════════════════════════════════════

def smeared_daily_from_dict_trades(trades: List[Dict], index: pd.DatetimeIndex
                                    ) -> pd.Series:
    s = pd.Series(0.0, index=index)
    for t in trades:
        try:
            entry = pd.Timestamp(t["entry_date"])
            exit_ = pd.Timestamp(t["exit_date"])
        except Exception:
            continue
        window = index[(index >= entry) & (index <= exit_)]
        if len(window) == 0:
            continue
        per_day = float(t["pnl"]) / CAPITAL / len(window)
        s.loc[window] += per_day
    return s


def smeared_daily_from_spread_trades(trades: List, index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(0.0, index=index)
    for t in trades:
        try:
            entry = pd.Timestamp(t.entry_date)
            exit_ = pd.Timestamp(t.expiration)
        except Exception:
            continue
        window = index[(index >= entry) & (index <= exit_)]
        if len(window) == 0:
            continue
        per_day = float(t.pnl_pct_capital) / len(window)
        s.loc[window] += per_day
    return s


def sparse_daily_from_dict_trades(trades: List[Dict], index: pd.DatetimeIndex
                                    ) -> pd.Series:
    if not trades:
        return pd.Series(0.0, index=index)
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    return daily.reindex(index, fill_value=0.0)


def sparse_daily_from_spread_trades(trades: List, index: pd.DatetimeIndex) -> pd.Series:
    if not trades:
        return pd.Series(0.0, index=index)
    data = [{"exit_date": t.expiration, "pnl": t.pnl_pct_capital * CAPITAL} for t in trades]
    df = pd.DataFrame(data)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    return daily.reindex(index, fill_value=0.0)


def trade_level_sharpe(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    pnls = np.array([t["pnl"] for t in trades])
    if len(pnls) < 2:
        return 0.0
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"])
    ex = pd.to_datetime(df["exit_date"])
    yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
    tpy = len(pnls) / yrs
    rets = pnls / CAPITAL
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    return float(mu / sd * math.sqrt(tpy)) if sd > 1e-12 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Build 9-stream dataframes (smeared + sparse)
# ═══════════════════════════════════════════════════════════════════════════

def build_streams() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    full_idx = pd.bdate_range(START, END)

    # Canonical cached streams
    with open(CACHE_V3, "rb") as fh:
        v3 = pickle.load(fh)
    gld_cal = v3["gld_calendar"].reindex(full_idx, fill_value=0.0)
    slv_cal = v3["slv_calendar"].reindex(full_idx, fill_value=0.0)
    v5_hedge = v3["v5_hedge"].reindex(full_idx, fill_value=0.0)

    # EXP-1220 / XLF / XLI / vol_arb (same as EXP-2200)
    with open(CACHE_EXP1220, "rb") as fh:
        exp1220_trades = pickle.load(fh)
    with open(CACHE_XLF, "rb") as fh:
        xlf_trades = pickle.load(fh)
    with open(CACHE_XLI, "rb") as fh:
        xli_trades = pickle.load(fh)
    with open(CACHE_VOL_ARB, "rb") as fh:
        vol_arb_trades = pickle.load(fh)

    # New: QQQ and EEM VRP
    qqq_trades = load_qqq_trades()
    eem_vrp = load_eem_vrp_stream().reindex(full_idx, fill_value=0.0)

    # SMEARED
    smeared = pd.DataFrame({
        "exp1220":  smeared_daily_from_dict_trades(exp1220_trades, full_idx),
        "xlf_cs":   smeared_daily_from_spread_trades(xlf_trades, full_idx),
        "xli_cs":   smeared_daily_from_spread_trades(xli_trades, full_idx),
        "qqq_cs":   smeared_daily_from_dict_trades(qqq_trades, full_idx),
        "gld_cal":  gld_cal,
        "slv_cal":  slv_cal,
        "vol_arb":  smeared_daily_from_dict_trades(vol_arb_trades, full_idx),
        "v5_hedge": v5_hedge,
        "eem_vrp":  eem_vrp,
    }).fillna(0.0)

    # SPARSE
    sparse = pd.DataFrame({
        "exp1220":  sparse_daily_from_dict_trades(exp1220_trades, full_idx),
        "xlf_cs":   sparse_daily_from_spread_trades(xlf_trades, full_idx),
        "xli_cs":   sparse_daily_from_spread_trades(xli_trades, full_idx),
        "qqq_cs":   sparse_daily_from_dict_trades(qqq_trades, full_idx),
        "gld_cal":  gld_cal,
        "slv_cal":  slv_cal,
        "vol_arb":  sparse_daily_from_dict_trades(vol_arb_trades, full_idx),
        "v5_hedge": v5_hedge,
        "eem_vrp":  eem_vrp,
    }).fillna(0.0)

    info: Dict[str, Dict] = {}
    info["exp1220"] = {"n_trades": len(exp1220_trades),
                       "trade_sharpe": round(trade_level_sharpe(exp1220_trades), 3)}
    info["qqq_cs"]  = {"n_trades": len(qqq_trades),
                       "trade_sharpe": round(trade_level_sharpe(qqq_trades), 3)}
    info["vol_arb"] = {"n_trades": len(vol_arb_trades),
                       "trade_sharpe": round(trade_level_sharpe(vol_arb_trades), 3)}
    info["xlf_cs"]  = {"n_trades": len(xlf_trades), "trade_sharpe": None}
    info["xli_cs"]  = {"n_trades": len(xli_trades), "trade_sharpe": None}
    info["gld_cal"] = {"n_trades": None, "trade_sharpe": None}
    info["slv_cal"] = {"n_trades": None, "trade_sharpe": None}
    info["vol_arb"]["trade_sharpe"] = info["vol_arb"]["trade_sharpe"]
    info["v5_hedge"] = {"n_trades": None, "trade_sharpe": None}
    info["eem_vrp"] = {"n_trades": None, "trade_sharpe": None}
    return smeared, sparse, info


# ═══════════════════════════════════════════════════════════════════════════
# 5. Equal-risk weights on a given sleeve list
# ═══════════════════════════════════════════════════════════════════════════

def equal_risk_weights(returns: pd.DataFrame, sleeves: List[str]) -> Dict[str, float]:
    vols = np.array([returns[k].std() + 1e-12 for k in sleeves])
    inv = 1.0 / vols
    w = inv / inv.sum()
    return {k: float(v) for k, v in zip(sleeves, w)}


def compose(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    port = pd.Series(0.0, index=streams.index)
    for k, w in weights.items():
        if k in streams.columns:
            port = port + w * streams[k]
    return port


def vol_target(port: pd.Series, target: float) -> pd.Series:
    realized = port.std() * math.sqrt(252)
    if realized < 1e-12:
        return port
    return port * (target / realized)


def yearly(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. HTML
# ═══════════════════════════════════════════════════════════════════════════

def build_html(payload: Dict) -> str:
    smeared = payload["stream_metrics_smeared"]
    info = payload["stream_info"]

    stream_rows = ""
    for k in payload["all_streams"]:
        m = smeared[k]
        inf = info.get(k, {})
        nt = inf.get("n_trades") or "—"
        ts = inf.get("trade_sharpe")
        ts_str = f"{ts:.2f}" if ts is not None else "—"
        stream_rows += (
            f"<tr><td>{k}</td><td>{nt}</td><td>{ts_str}</td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['vol_pct']:.2f}%</td></tr>"
        )

    cfg_rows = ""
    for label, cfg in payload["configs"].items():
        m = cfg["metrics_smeared"]
        sp = cfg["metrics_sparse"]
        marker = " ★" if label == payload["winner"] else ""
        lift = sp["sharpe"] - payload["baseline_sparse_sharpe"]
        lift_color = "#16a34a" if lift > 0 else ("#dc2626" if lift < 0 else "#0f172a")
        pass_color = "#16a34a" if sp["sharpe"] >= 6.0 else "#0f172a"
        cfg_rows += (
            f"<tr><td><strong>{label}{marker}</strong></td>"
            f"<td>{cfg['n_streams']}</td>"
            f"<td>{m['cagr_pct']:.0f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td style='font-weight:700;color:{pass_color}'>{sp['sharpe']:.2f}</td>"
            f"<td style='color:{lift_color};font-weight:700'>{lift:+.2f}</td>"
            f"<td>{sp['max_dd_pct']:.1f}%</td>"
            f"<td>{cfg.get('vol_target') or '—'}</td></tr>"
        )

    corr = pd.DataFrame(payload["correlation_matrix"])
    corr_rows = ""
    for ix in corr.index:
        cells = ""
        for cx in corr.columns:
            v = corr.loc[ix, cx]
            color = "#16a34a" if v < 0 else ("#dc2626" if v > 0.5 else "#0f172a")
            cells += f"<td style='color:{color}'>{v:+.3f}</td>"
        corr_rows += f"<tr><td>{ix}</td>{cells}</tr>"

    winner = payload["winner"]
    wcfg = payload["configs"][winner]
    wm = wcfg["metrics_sparse"]
    pass_target = "YES ✓" if wm["sharpe"] >= 6.0 else "NO"
    verdict_color = "#16a34a" if wm["sharpe"] >= 6.0 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2250 — North Star v7 (9 streams)</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1300px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.85em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.winner {{ background:#fff;border:2px solid {verdict_color};border-radius:10px;padding:18px;margin:20px 0; }}
.winner h3 {{ margin-top:0;color:{verdict_color}; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.84em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2250 — 8/9-Stream North Star Portfolio v7</h1>
<p style="color:#64748b">Adding QQQ credit spreads + EEM VRP to the 7-stream v6 ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — all real:</strong><br>
Carried over from EXP-2200 (7 streams): exp1220, xlf_cs, xli_cs, gld_cal,
slv_cal, vol_arb, v5_hedge<br>
NEW stream 8 — qqq_cs: compass.exp2240_qqq_iwm_credit_spreads
.run_credit_spread_trades on real IronVault QQQ chains + Yahoo QQQ/^VIX<br>
NEW stream 9 — eem_vrp: compass.exp1660_vrp_deepening walk-forward on real
Yahoo EEM + FRED VXEEMCLS, vol-targeted to 5%/yr<br>
Sharpe: compass.metrics.full_metrics (canonical mean/std × √252)
</div>

<div class="winner">
<h3>★ Winner: <code>{winner}</code> — 6.0 target {pass_target}</h3>
<strong>SPARSE Sharpe (honest):</strong> {wm['sharpe']:.2f} ·
CAGR {wm['cagr_pct']:.1f}% · Max DD {wm['max_dd_pct']:.1f}% ·
Calmar {wm['calmar']:.2f}<br>
Baseline (EXP-2200 7-stream equal_risk_15%): {payload['baseline_sparse_sharpe']:.2f}<br>
Lift from adding new streams: <strong>{wm['sharpe'] - payload['baseline_sparse_sharpe']:+.2f}</strong>
</div>

<h2>1. Stream-level metrics (smeared, all 9 streams)</h2>
<table>
<thead><tr><th>Stream</th><th>N trades</th><th>Trade SR</th><th>CAGR</th><th>Smeared SR</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>{stream_rows}</tbody>
</table>

<h2>2. Correlation matrix (smeared daily)</h2>
<table>
<thead><tr><th></th>{''.join(f'<th>{c}</th>' for c in corr.columns)}</tr></thead>
<tbody>{corr_rows}</tbody>
</table>

<h2>3. Portfolio configurations — equal-risk weights, vol targets</h2>
<table>
<thead><tr>
<th>Config</th><th>N sleeves</th>
<th>Smeared CAGR</th><th>Smeared SR</th>
<th>Sparse SR (honest)</th><th>ΔSR vs baseline</th>
<th>Sparse DD</th><th>Target vol</th>
</tr></thead>
<tbody>{cfg_rows}</tbody>
</table>

<div class="note">
<strong>Methodology:</strong> smeared daily Sharpe uses EXP-2160's uniform
P&L smearing across entry→exit (dense stream, upper bound). Sparse
daily Sharpe uses exit-date only (conservative lower bound). The
honest portfolio Sharpe sits between the two. We select the winner by
SPARSE Sharpe because that's the most-conservative honest measurement.
Baseline for ΔSR is the EXP-2200 winning config (7-stream equal_risk_15%
sparse Sharpe 5.96).
</div>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2250_north_star_v7.py · Rule Zero · all real data
</p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2250 — 8/9-Stream North Star Portfolio v7")
    print("=" * 72)

    smeared, sparse, info = build_streams()

    all_streams = list(smeared.columns)
    print(f"\n[streams] {len(all_streams)}: {all_streams}")

    # Standalone metrics
    smeared_metrics = {k: full_metrics(smeared[k].values) for k in all_streams}
    print("\n[standalone smeared metrics]:")
    for k in all_streams:
        m = smeared_metrics[k]
        ts = info[k].get("trade_sharpe")
        ts_str = f"  trade_SR {ts:.2f}" if ts is not None else ""
        print(f"  {k:10s}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:6.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Vol {m['vol_pct']:5.1f}%{ts_str}")

    corr = smeared[all_streams].corr().round(3)
    print("\n[correlation matrix]:")
    print(corr.to_string())

    # Configurations
    # v6 baseline (7 streams) + QQQ (8) + QQQ+EEM (9)
    configs: Dict[str, Dict] = {}

    sleeve_sets = {
        "v6_7stream":    ["exp1220", "xlf_cs", "xli_cs", "gld_cal", "slv_cal", "vol_arb", "v5_hedge"],
        "v7_8stream_QQQ": ["exp1220", "xlf_cs", "xli_cs", "qqq_cs", "gld_cal", "slv_cal", "vol_arb", "v5_hedge"],
        "v7_9stream_QQQ_EEM": ["exp1220", "xlf_cs", "xli_cs", "qqq_cs", "gld_cal", "slv_cal", "vol_arb", "v5_hedge", "eem_vrp"],
    }
    vol_targets = [(None, "none"), (0.12, "12%"), (0.15, "15%")]

    print("\n[configs] equal-risk × vol targets:")
    for set_name, sleeves in sleeve_sets.items():
        w = equal_risk_weights(smeared, sleeves)
        for tv, tv_label in vol_targets:
            label = f"{set_name}_{tv_label}"
            port_sm = compose(smeared, w)
            port_sp = compose(sparse, w)
            if tv is not None:
                port_sm = vol_target(port_sm, tv)
                port_sp = vol_target(port_sp, tv)
            m_sm = full_metrics(port_sm.iloc[WARMUP:].values)
            m_sp = full_metrics(port_sp.iloc[WARMUP:].values)
            configs[label] = {
                "n_streams": len(sleeves),
                "weights": w,
                "vol_target": tv_label if tv else None,
                "metrics_smeared": m_sm,
                "metrics_sparse": m_sp,
                "yearly_sparse": yearly(port_sp.iloc[WARMUP:]),
            }
            print(f"  {label:32s} (n={len(sleeves)})  "
                  f"smSR {m_sm['sharpe']:6.2f}  "
                  f"spSR {m_sp['sharpe']:5.2f}  "
                  f"spCAGR {m_sp['cagr_pct']:+6.1f}%  "
                  f"spDD {m_sp['max_dd_pct']:4.1f}%")

    # Winner by sparse Sharpe
    winner = max(configs.keys(),
                 key=lambda k: configs[k]["metrics_sparse"]["sharpe"])
    baseline_sparse = configs["v6_7stream_15%"]["metrics_sparse"]["sharpe"]
    w_sp = configs[winner]["metrics_sparse"]["sharpe"]
    print(f"\n[winner] {winner} — sparse Sharpe {w_sp:.2f}")
    print(f"         baseline v6 7stream_15% sparse: {baseline_sparse:.2f}")
    print(f"         lift: {w_sp - baseline_sparse:+.2f}")
    print(f"         6.0 target: {'MET' if w_sp >= 6.0 else 'MISSED (gap ' + f'{6.0 - w_sp:+.2f})'}")

    # JSON
    payload = {
        "experiment": "EXP-2250",
        "title": "8/9-Stream North Star Portfolio v7",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "base_7_streams": "exp2200 cache reuse (exp1220, xlf_cs, xli_cs, gld_cal, slv_cal, vol_arb, v5_hedge)",
            "qqq_cs": "compass.exp2240_qqq_iwm_credit_spreads on real IronVault QQQ chains",
            "eem_vrp": "compass.exp1660_vrp_deepening walk-forward on real Yahoo EEM + FRED VXEEMCLS, vol-targeted 5%/yr",
            "sharpe_formula": "compass.metrics.full_metrics (mean/std × √252)",
            "trade_to_daily": "uniform smearing (EXP-2160 method) for upper bound, exit-date for sparse lower bound",
        },
        "data_window": {"start": START, "end": END, "warmup": WARMUP},
        "all_streams": all_streams,
        "stream_info": info,
        "stream_metrics_smeared": smeared_metrics,
        "correlation_matrix": corr.to_dict(),
        "sleeve_sets": sleeve_sets,
        "configs": configs,
        "winner": winner,
        "winner_sparse_sharpe": round(w_sp, 3),
        "baseline_sparse_sharpe": round(baseline_sparse, 3),
        "lift_vs_baseline": round(w_sp - baseline_sparse, 3),
        "six_point_oh_target_met": w_sp >= 6.0,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
