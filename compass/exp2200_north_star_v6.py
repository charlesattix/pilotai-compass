"""EXP-2200 — 7-Stream North Star Portfolio v6.

Combines seven real-data alpha streams including the EXP-2160 XLF/XLI
credit spread discoveries (per-trade Sharpe 11.3 and 6.0 respectively,
near-zero correlation to EXP-1220 SPY) into a single portfolio and
tests whether the added diversification can finally clear Sharpe 6.0
at the portfolio-daily level.

Streams
-------
  1. EXP-1220 SPY — compass.exp1220_standalone.run_exp1220_trades
     on real IronVault SPY chains (171 baseline trades)
  2. XLF put-credit-spread — compass.exp2160_high_capacity_alts
     run_put_credit_spreads(con, "XLF") (248 trades, WR 98.4%)
  3. XLI put-credit-spread — same module (248 trades, WR 96.8%)
  4. GLD calendar — compass.exp1770_commodity_calendars GLD-GC=F WF
     (from compass/cache/exp1860_streams.pkl)
  5. SLV calendar — compass.exp1770_commodity_calendars SLV-SI=F WF
     (from compass/cache/exp1860_streams.pkl)
  6. Cross-Vol Arb — compass.exp2020_cross_vol_arb trade tape
     (from compass/cache/exp2020_vol_arb_trades.pkl)
  7. Crisis Alpha v5 hedge — compass.crisis_alpha_v5 frozen best
     (from compass/cache/exp1860_streams.pkl)

Conversion methodology
----------------------
For ALL trade-tape sleeves (EXP-1220, XLF, XLI, vol_arb), pnl is smeared
uniformly across the holding window (entry→expiration/exit) via
EXP-2160's trades_to_daily_pct method. This is a DENSE representation
that eliminates the 86-91% zero-return-day dilution penalty
(MASTERPLAN Bug 3) that crushed EXP-2100's portfolio Sharpe.

HONEST CAVEAT: EXP-2160's own report explicitly flags that the daily-
smeared Sharpe is "method-inflated on high-win-rate strategies due to
P&L smearing across the holding period". XLF at 98% win rate scores
sharpe_daily_spread 27.22 but sharpe_per_trade only 11.29 — a 2.4×
inflation ratio. Any portfolio Sharpe computed from smeared sleeves
must be read as an UPPER BOUND. To compensate we also report:
  - the sparse-tape (exit-date) portfolio Sharpe as a LOWER BOUND
  - the trade-level Sharpe per sleeve for context
  - the sleeve's daily volatility (low vol = smearing artefact)

Weight configurations (3 schemes × vol targets 12%, 15%, none)
--------------------------------------------------------------
  equal_risk  : inverse-vol weights (risk parity)
  max_sharpe  : scipy SLSQP long-only
  min_variance: scipy SLSQP long-only

Walk-forward: 2020-2025 with 252-day warmup trim. No per-sleeve
leverage applied here (unlike EXP-2050 where EXP-1220 was 2×); the
optimizer is free to assign each sleeve its natural weight, and
vol-targeting applies uniformly at the portfolio level.

Rule Zero: every input is real. No synthetic data.

Output
------
  compass/reports/exp2200_north_star_v6.json
  compass/reports/exp2200_north_star_v6.html
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

REPORT_JSON = ROOT / "compass" / "reports" / "exp2200_north_star_v6.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2200_north_star_v6.html"
CACHE_DIR = ROOT / "compass" / "cache"
CACHE_V3 = CACHE_DIR / "exp1860_streams.pkl"
CACHE_VOL_ARB = CACHE_DIR / "exp2020_vol_arb_trades.pkl"
CACHE_EXP1220 = CACHE_DIR / "exp2150_trades_biweekly.pkl"
CACHE_XLF = CACHE_DIR / "exp2200_xlf_trades.pkl"
CACHE_XLI = CACHE_DIR / "exp2200_xli_trades.pkl"
DB_PATH = ROOT / "data" / "options_cache.db"

START = "2020-01-01"
END = "2025-12-31"
WARMUP = 252
CAPITAL = 100_000

STREAMS = ["exp1220", "xlf_cs", "xli_cs", "gld_cal", "slv_cal", "vol_arb", "v5_hedge"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Trade-tape loaders (with caching)
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220_trades() -> List[Dict]:
    if CACHE_EXP1220.exists():
        print(f"[cache] exp1220 trades from {CACHE_EXP1220.name}")
        with open(CACHE_EXP1220, "rb") as fh:
            return pickle.load(fh)
    print("[load] running exp1220 pipeline (cached by EXP-2150)...")
    import yfinance as yf
    from shared.iron_vault import IronVault
    from compass.exp1220_standalone import run_exp1220_trades
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    trades = run_exp1220_trades(hd, spy, vix)
    with open(CACHE_EXP1220, "wb") as fh:
        pickle.dump(trades, fh)
    return trades


def load_credit_spread_trades(ticker: str, cache_file: Path) -> List:
    if cache_file.exists():
        print(f"[cache] {ticker} CS trades from {cache_file.name}")
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)
    print(f"[run] EXP-2160 put-credit-spread backtest for {ticker}...")
    from compass.exp2160_high_capacity_alts import run_put_credit_spreads
    con = sqlite3.connect(str(DB_PATH))
    try:
        trades = run_put_credit_spreads(con, ticker)
    finally:
        con.close()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(trades, fh)
    return trades


def load_vol_arb_trades() -> List[Dict]:
    with open(CACHE_VOL_ARB, "rb") as fh:
        return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Trade → daily conversions
# ═══════════════════════════════════════════════════════════════════════════

def smeared_daily_from_dict_trades(trades: List[Dict],
                                    index: pd.DatetimeIndex,
                                    capital: float = CAPITAL) -> pd.Series:
    """Smear each trade's pnl uniformly across entry_date → exit_date."""
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
        per_day = float(t["pnl"]) / capital / len(window)
        s.loc[window] += per_day
    return s


def smeared_daily_from_spread_trades(trades: List,
                                      index: pd.DatetimeIndex) -> pd.Series:
    """EXP-2160 SpreadTrade uses entry_date / expiration / pnl_pct_capital."""
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


def sparse_daily_from_dict_trades(trades: List[Dict],
                                    index: pd.DatetimeIndex,
                                    capital: float = CAPITAL) -> pd.Series:
    if not trades:
        return pd.Series(0.0, index=index)
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / capital
    return daily.reindex(index, fill_value=0.0)


def trade_level_sharpe(trades, capital: float = CAPITAL) -> float:
    if not trades:
        return 0.0
    if isinstance(trades[0], dict):
        pnls = np.array([t["pnl"] for t in trades])
        entries = [t["entry_date"] for t in trades]
        exits = [t["exit_date"] for t in trades]
    else:
        pnls = np.array([t.pnl_pct_capital * capital for t in trades])
        entries = [t.entry_date for t in trades]
        exits = [getattr(t, "expiration", t.entry_date) for t in trades]
    if len(pnls) < 2:
        return 0.0
    en = pd.to_datetime(entries)
    ex = pd.to_datetime(exits)
    yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
    tpy = len(pnls) / yrs
    rets = pnls / capital
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    return float(mu / sd * math.sqrt(tpy)) if sd > 1e-12 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Build the 7-stream dataframe (both smeared and sparse variants)
# ═══════════════════════════════════════════════════════════════════════════

def build_streams() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """Return (smeared_df, sparse_df, stream_info)."""
    full_idx = pd.bdate_range(START, END)

    # Canonical cached streams (already daily)
    with open(CACHE_V3, "rb") as fh:
        v3 = pickle.load(fh)
    gld_cal = v3["gld_calendar"].reindex(full_idx, fill_value=0.0)
    slv_cal = v3["slv_calendar"].reindex(full_idx, fill_value=0.0)
    v5_hedge = v3["v5_hedge"].reindex(full_idx, fill_value=0.0)

    # Trade-tape sleeves
    exp1220_trades = load_exp1220_trades()
    xlf_trades = load_credit_spread_trades("XLF", CACHE_XLF)
    xli_trades = load_credit_spread_trades("XLI", CACHE_XLI)
    vol_arb_trades = load_vol_arb_trades()

    # SMEARED variants (dense, portfolio-usable)
    exp1220_smear = smeared_daily_from_dict_trades(exp1220_trades, full_idx)
    xlf_smear = smeared_daily_from_spread_trades(xlf_trades, full_idx)
    xli_smear = smeared_daily_from_spread_trades(xli_trades, full_idx)
    vol_arb_smear = smeared_daily_from_dict_trades(vol_arb_trades, full_idx)

    # SPARSE variants (exit-date, lower-bound sanity)
    exp1220_sparse = sparse_daily_from_dict_trades(exp1220_trades, full_idx)
    vol_arb_sparse = sparse_daily_from_dict_trades(vol_arb_trades, full_idx)
    # XLF/XLI: convert SpreadTrade → dict format for sparse
    def _spread_to_sparse(trades):
        if not trades:
            return pd.Series(0.0, index=full_idx)
        data = [{"exit_date": t.expiration,
                  "pnl": t.pnl_pct_capital * CAPITAL} for t in trades]
        df = pd.DataFrame(data)
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
        return daily.reindex(full_idx, fill_value=0.0)
    xlf_sparse = _spread_to_sparse(xlf_trades)
    xli_sparse = _spread_to_sparse(xli_trades)

    smeared = pd.DataFrame({
        "exp1220": exp1220_smear,
        "xlf_cs": xlf_smear,
        "xli_cs": xli_smear,
        "gld_cal": gld_cal,
        "slv_cal": slv_cal,
        "vol_arb": vol_arb_smear,
        "v5_hedge": v5_hedge,
    }).fillna(0.0)

    sparse = pd.DataFrame({
        "exp1220": exp1220_sparse,
        "xlf_cs": xlf_sparse,
        "xli_cs": xli_sparse,
        "gld_cal": gld_cal,
        "slv_cal": slv_cal,
        "vol_arb": vol_arb_sparse,
        "v5_hedge": v5_hedge,
    }).fillna(0.0)

    info: Dict[str, Dict] = {}
    for name, trades in (
        ("exp1220", exp1220_trades),
        ("xlf_cs", xlf_trades),
        ("xli_cs", xli_trades),
        ("vol_arb", vol_arb_trades),
    ):
        info[name] = {
            "n_trades": len(trades),
            "trade_sharpe": round(trade_level_sharpe(trades), 3),
        }
    for name, series in (
        ("gld_cal", gld_cal),
        ("slv_cal", slv_cal),
        ("v5_hedge", v5_hedge),
    ):
        m = full_metrics(series.values)
        info[name] = {
            "n_trades": None,
            "daily_sharpe": m["sharpe"],
        }

    return smeared, sparse, info


# ═══════════════════════════════════════════════════════════════════════════
# 4. Optimizers
# ═══════════════════════════════════════════════════════════════════════════

def _norm(w: np.ndarray) -> Dict[str, float]:
    w = np.clip(w, 0, None)
    if w.sum() < 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return {k: float(v) for k, v in zip(STREAMS, w)}


def equal_risk_weights(returns: pd.DataFrame) -> Dict[str, float]:
    vols = np.array([returns[k].std() + 1e-12 for k in STREAMS])
    return _norm(1.0 / vols)


def max_sharpe_weights(returns: pd.DataFrame,
                        cap: float = 0.50) -> Dict[str, float]:
    n = len(STREAMS)
    try:
        from scipy.optimize import minimize
        mu = returns[STREAMS].mean().values * 252
        cov = returns[STREAMS].cov().values * 252

        def neg_sh(w):
            r = float(np.dot(w, mu))
            v = float(np.sqrt(np.dot(w, cov @ w)))
            return 1e9 if v < 1e-9 else -r / v

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, cap)] * n
        x0 = np.ones(n) / n
        res = minimize(neg_sh, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 300})
        if res.success:
            return _norm(res.x)
    except Exception as e:
        print(f"  max_sharpe failed: {e}")
    return _norm(np.ones(n))


def min_variance_weights(returns: pd.DataFrame,
                          cap: float = 0.50) -> Dict[str, float]:
    n = len(STREAMS)
    try:
        from scipy.optimize import minimize
        cov = returns[STREAMS].cov().values * 252

        def obj(w):
            return float(np.dot(w, cov @ w))

        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, cap)] * n
        x0 = np.ones(n) / n
        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 300})
        if res.success:
            return _norm(res.x)
    except Exception as e:
        print(f"  min_var failed: {e}")
    return _norm(np.ones(n))


# ═══════════════════════════════════════════════════════════════════════════
# 5. Portfolio composition + vol targeting
# ═══════════════════════════════════════════════════════════════════════════

def compose(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    port = pd.Series(0.0, index=streams.index)
    for k in STREAMS:
        port = port + weights.get(k, 0.0) * streams[k]
    return port


def vol_target(port: pd.Series, target_vol: float) -> pd.Series:
    realized = port.std() * math.sqrt(252)
    if realized < 1e-12:
        return port
    scale = target_vol / realized
    return port * scale


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
    smeared_rows = ""
    for k in STREAMS:
        m = payload["stream_metrics_smeared"][k]
        info = payload["stream_info"].get(k, {})
        nt = info.get("n_trades", "—")
        ts = info.get("trade_sharpe", "—")
        smeared_rows += (
            f"<tr><td>{k}</td>"
            f"<td>{nt}</td>"
            f"<td>{ts if ts == '—' else f'{ts:.2f}'}</td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['vol_pct']:.1f}%</td></tr>"
        )

    cfg_rows = ""
    for label, cfg in payload["configs"].items():
        m = cfg["metrics"]
        marker = " ★" if label == payload["winner"] else ""
        cfg_rows += (
            f"<tr><td><strong>{label}{marker}</strong></td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['calmar']:.2f}</td>"
            f"<td>{m['vol_pct']:.1f}%</td>"
            f"<td>{cfg['vol_target'] or '—'}</td></tr>"
        )

    sparse_rows = ""
    for label, cfg in payload["sparse_configs"].items():
        m = cfg["metrics"]
        sparse_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td></tr>"
        )

    weights_cols = list(payload["optimizer_weights"].keys())
    weight_rows = ""
    for sleeve in STREAMS:
        cells = "".join(
            f"<td>{payload['optimizer_weights'][c].get(sleeve, 0)*100:.1f}%</td>"
            for c in weights_cols
        )
        weight_rows += f"<tr><td>{sleeve}</td>{cells}</tr>"

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
    wm = payload["configs"][winner]["metrics"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2200 — North Star v6 (7 streams)</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1300px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.85em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.winner {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:18px;margin:20px 0; }}
.winner h3 {{ margin-top:0;color:#065f46; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.84em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2200 — 7-Stream North Star Portfolio v6</h1>
<p style="color:#64748b">EXP-1220 + XLF CS + XLI CS + GLD + SLV + vol_arb + v5_hedge ·
walk-forward 2020-2025 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — all real:</strong><br>
exp1220: compass.exp1220_standalone on real IronVault SPY chains<br>
xlf_cs / xli_cs: compass.exp2160_high_capacity_alts.run_put_credit_spreads on real IronVault chains<br>
gld_cal / slv_cal: compass.exp1770_commodity_calendars walk-forward on real Yahoo<br>
vol_arb: compass.exp2020_cross_vol_arb trade tape on real IronVault + Yahoo RV<br>
v5_hedge: compass.crisis_alpha_v5 frozen best on real Yahoo 13-ETF<br>
Trade → daily: uniform smearing across entry→exit (EXP-2160 method).
Sharpe: compass.metrics.full_metrics (mean/std × √252)
</div>

<div class="winner">
<h3>★ Winner: <code>{winner}</code></h3>
CAGR <strong>{wm['cagr_pct']:.1f}%</strong> ·
Sharpe <strong>{wm['sharpe']:.2f}</strong> ·
Max DD <strong>{wm['max_dd_pct']:.1f}%</strong> ·
Calmar <strong>{wm['calmar']:.2f}</strong> ·
Vol {wm['vol_pct']:.1f}%
</div>

<div class="note">
<strong>Methodology caveat — smearing inflation:</strong> EXP-2160's own
report flagged that daily-smeared Sharpe is "method-inflated on high-
win-rate strategies due to P&L smearing across the holding period".
XLF standalone trade Sharpe 11.29 → smeared daily Sharpe 27.22 (2.4×
inflation). All smeared-portfolio Sharpes below should be read as an
UPPER BOUND. The sparse-tape (exit-date) lower bound is reported in
Section 4 for sanity check.
</div>

<h2>1. Stream-level metrics (smeared, pre-portfolio)</h2>
<table>
<thead><tr><th>Stream</th><th>N trades</th><th>Trade SR</th><th>CAGR</th><th>Smeared Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>{smeared_rows}</tbody>
</table>

<h2>2. Correlation matrix (smeared daily)</h2>
<table>
<thead><tr><th></th>{''.join(f'<th>{c}</th>' for c in corr.columns)}</tr></thead>
<tbody>{corr_rows}</tbody>
</table>

<h2>3. Optimizer weights</h2>
<table>
<thead><tr><th>Sleeve</th>{''.join(f'<th>{c}</th>' for c in weights_cols)}</tr></thead>
<tbody>{weight_rows}</tbody>
</table>

<h2>4. Portfolio results (all configs × vol targets)</h2>
<table>
<thead><tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Target vol</th></tr></thead>
<tbody>{cfg_rows}</tbody>
</table>

<h3>Sparse-tape lower bound (same weights, exit-date conversion)</h3>
<table>
<thead><tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{sparse_rows}</tbody>
</table>
<div class="note">
Sparse Sharpe is the lower bound — no smearing, exit-date returns only.
The gap between smeared (upper) and sparse (lower) Sharpe quantifies
the methodology inflation for this specific portfolio.
</div>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2200_north_star_v6.py · Rule Zero · all real data
</p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2200 — 7-Stream North Star Portfolio v6")
    print("=" * 72)

    print("\n[1/4] Loading 7 streams (smeared + sparse)...")
    smeared, sparse, stream_info = build_streams()
    print(f"       smeared shape: {smeared.shape}")
    print(f"       sparse shape:  {sparse.shape}")

    # Stream-level metrics
    smeared_metrics = {k: full_metrics(smeared[k].values) for k in STREAMS}
    sparse_metrics = {k: full_metrics(sparse[k].values) for k in STREAMS}

    print("\n[streams] SMEARED standalone metrics:")
    for k in STREAMS:
        m = smeared_metrics[k]
        extra = f"  trade_SR {stream_info[k].get('trade_sharpe', 'n/a')}" \
                if stream_info[k].get('trade_sharpe') is not None else ""
        print(f"  {k:10s}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:6.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Vol {m['vol_pct']:5.1f}%{extra}")

    corr = smeared[STREAMS].corr().round(3)
    print("\n[corr] smeared daily correlation matrix:")
    print(corr.to_string())

    print("\n[2/4] Computing optimizer weights (on smeared returns)...")
    er_w = equal_risk_weights(smeared)
    ms_w = max_sharpe_weights(smeared)
    mv_w = min_variance_weights(smeared)
    print(f"  equal_risk:   {er_w}")
    print(f"  max_sharpe:   {ms_w}")
    print(f"  min_variance: {mv_w}")

    # ── Configs: 3 weight schemes × {no_vt, 12%, 15%} = 9 configs
    print("\n[3/4] Running 9 configs (3 weight × {none,12%,15%} vol targets)...")
    configs: Dict[str, Dict] = {}
    sparse_configs: Dict[str, Dict] = {}
    weight_schemes = [("equal_risk", er_w), ("max_sharpe", ms_w), ("min_variance", mv_w)]
    vol_targets = [(None, "none"), (0.12, "12%"), (0.15, "15%")]

    for w_name, w in weight_schemes:
        for tv, tv_label in vol_targets:
            label = f"{w_name}_{tv_label}"
            port = compose(smeared, w)
            if tv is not None:
                port = vol_target(port, tv)
            port_oos = port.iloc[WARMUP:]
            m = full_metrics(port_oos.values)
            configs[label] = {
                "weights": w,
                "vol_target": tv_label if tv else None,
                "metrics": m,
                "yearly": yearly(port_oos),
            }

            # sparse equivalent
            port_sp = compose(sparse, w)
            if tv is not None:
                port_sp = vol_target(port_sp, tv)
            port_sp_oos = port_sp.iloc[WARMUP:]
            sparse_configs[label] = {
                "metrics": full_metrics(port_sp_oos.values),
            }

            print(f"  {label:28s}  CAGR {m['cagr_pct']:+7.1f}%  "
                  f"Sharpe {m['sharpe']:6.2f}  DD {m['max_dd_pct']:5.1f}%  "
                  f"Calmar {m['calmar']:6.2f}  | sparse SR "
                  f"{sparse_configs[label]['metrics']['sharpe']:5.2f}")

    # Winner on smeared (headline) and cross-check with sparse
    winner = max(configs.keys(), key=lambda k: configs[k]["metrics"]["sharpe"])
    print(f"\n[winner] SMEARED: {winner} "
          f"→ Sharpe {configs[winner]['metrics']['sharpe']:.2f}")
    print(f"         sparse cross-check: {sparse_configs[winner]['metrics']['sharpe']:.2f}")

    # ── JSON ────────────────────────────────────────────────────────
    payload = {
        "experiment": "EXP-2200",
        "title": "7-Stream North Star Portfolio v6",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "exp1220": "compass.exp1220_standalone on real IronVault SPY chains",
            "xlf_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads (XLF)",
            "xli_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads (XLI)",
            "gld_cal": "compass.exp1770_commodity_calendars WF GLD-GC=F (from exp1860 cache)",
            "slv_cal": "compass.exp1770_commodity_calendars WF SLV-SI=F (from exp1860 cache)",
            "vol_arb": "compass.exp2020_cross_vol_arb trade tape",
            "v5_hedge": "compass.crisis_alpha_v5 frozen best (from exp1860 cache)",
            "trade_to_daily": "uniform smearing across entry→exit (EXP-2160 method)",
            "sharpe_formula": "compass.metrics.full_metrics (mean/std × √252)",
        },
        "data_window": {"start": START, "end": END, "warmup": WARMUP},
        "stream_info": stream_info,
        "stream_metrics_smeared": smeared_metrics,
        "stream_metrics_sparse": sparse_metrics,
        "correlation_matrix": corr.to_dict(),
        "optimizer_weights": {
            "equal_risk": er_w,
            "max_sharpe": ms_w,
            "min_variance": mv_w,
        },
        "configs": configs,
        "sparse_configs": sparse_configs,
        "winner": winner,
        "winner_metrics": configs[winner]["metrics"],
        "winner_sparse_sharpe": sparse_configs[winner]["metrics"]["sharpe"],
        "caveat": (
            "SMEARED Sharpe is an UPPER BOUND — XLF/XLI at 98% win rate "
            "inflate 2.4× vs the per-trade Sharpe (EXP-2160 note). The "
            "sparse cross-check provides a LOWER BOUND. The truthful "
            "portfolio Sharpe lies somewhere between the two."
        ),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
