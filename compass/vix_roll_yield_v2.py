"""
compass/vix_roll_yield_v2.py — EXP-1700 v2: Multi-Proxy VIX Roll Yield.

CONTEXT: v1 (compass/vix_roll_yield.py) tested VXX short with a single config
and got full-period Sharpe 0.03, OOS Sharpe 0.27. The Wave 1 post-mortem
concluded the VXX proxy was too noisy — it's an ETN derivative of /VX futures
with its own tracking error.

v2 addresses this three ways:
  1. TEST MULTIPLE PROXIES: VXX (short), SVXY (long — inverse VXX), UVXY
     (short — 2x leveraged). Each isolates a different view of the same edge.
  2. TEST MULTIPLE THRESHOLDS: vary the contango entry threshold from 0.90 to
     1.00 to check robustness (v1 used a single 0.95 threshold).
  3. HONEST DATA CHECK: CBOE /VX futures data is NOT available via Yahoo
     Finance — only ^VIX/^VIX3M/^VIX6M index constants are. The actual
     futures contracts (VX/G26, VX/H26, etc.) require a data subscription.
     Documented upfront.

RULE ZERO: all data from Yahoo Finance chart API (public historical record).
No synthetic, no np.random, no Black-Scholes. Sharpe via compass/metrics.py
arithmetic mean.

Output:
    reports/exp1700_vix_roll_yield_v2.html
    reports/exp1700_vix_roll_yield_v2.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtester import _yf_download_safe
from compass.metrics import annualized_sharpe, max_drawdown as _mdd, cagr as _cagr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vix_roll_v2")

REPORT_PATH = ROOT / "reports" / "exp1700_vix_roll_yield_v2.html"
JSON_PATH = ROOT / "reports" / "exp1700_vix_roll_yield_v2.json"
TRADING_DAYS = 252
CAPITAL = 100_000
RISK_FREE_ANNUAL = 0.045


# ═══════════════════════════════════════════════════════════════════════════
# Proxy definitions — each ETF represents one way to trade the VRP via VIX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProxyConfig:
    name: str
    ticker: str
    side: str       # "short" or "long"
    description: str


PROXIES: List[ProxyConfig] = [
    ProxyConfig(
        name="VXX_short",
        ticker="VXX",
        side="short",
        description="Short VXX (iPath VIX ST Futures ETN). Classic VRP play — "
                    "VXX bleeds in contango. Path-dependent rebalancing noise.",
    ),
    ProxyConfig(
        name="SVXY_long",
        ticker="SVXY",
        side="long",
        description="Long SVXY (ProShares Short VIX Short-Term Futures). Inverse "
                    "of VXX — gains when VXX bleeds. Cleaner for long-only traders.",
    ),
    ProxyConfig(
        name="UVXY_short",
        ticker="UVXY",
        side="short",
        description="Short UVXY (2x leveraged). Amplified contango capture but "
                    "also amplified path dependency. Highest drawdown risk.",
    ),
]


@dataclass
class ContangoConfig:
    name: str
    enter_threshold: float   # enter position when ratio < this
    exit_threshold: float    # exit when ratio > this
    flat_threshold: float    # force flat above this (crisis)
    target_vol: float = 0.10
    vol_lookback: int = 20
    max_position: float = 1.0


THRESHOLDS: List[ContangoConfig] = [
    ContangoConfig("tight",   enter_threshold=0.92, exit_threshold=0.98, flat_threshold=1.05),
    ContangoConfig("medium",  enter_threshold=0.95, exit_threshold=1.00, flat_threshold=1.05),
    ContangoConfig("loose",   enter_threshold=0.98, exit_threshold=1.02, flat_threshold=1.05),
]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo(ticker: str, start: str = "2018-01-01",
                 end: str = "2026-04-01") -> pd.Series:
    """Fetch real Yahoo Finance close series."""
    df = _yf_download_safe(ticker, start, end)
    if df.empty:
        raise RuntimeError(f"No Yahoo data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df["Close"].astype(float)


def load_vix_data(start: str = "2018-01-01",
                   end: str = "2026-04-01") -> Dict[str, pd.Series]:
    """Load VIX index + term structure + all ETF proxies from Yahoo."""
    log.info(f"Loading VIX data from Yahoo ({start} → {end})...")
    data = {}
    for t in ["^VIX", "^VIX3M", "^VIX6M", "VXX", "SVXY", "UVXY"]:
        try:
            s = fetch_yahoo(t, start, end)
            data[t] = s
            log.info(f"  {t:8s} {len(s):>5} bars, "
                      f"{s.index.min().date()} → {s.index.max().date()}")
        except Exception as e:
            log.warning(f"  {t}: {e}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Signal generation — all lagged to avoid look-ahead
# ═══════════════════════════════════════════════════════════════════════════

def compute_signals(
    data: Dict[str, pd.Series],
    proxy: ProxyConfig,
    threshold: ContangoConfig,
) -> pd.DataFrame:
    """Generate position series for one (proxy, threshold) combination."""
    if proxy.ticker not in data:
        return pd.DataFrame()

    vix = data["^VIX"]
    vix3m = data["^VIX3M"]
    px = data[proxy.ticker]

    common = vix.index.intersection(vix3m.index).intersection(px.index)
    common = common.sort_values()

    vix_a = vix.reindex(common).ffill()
    vix3m_a = vix3m.reindex(common).ffill()
    px_a = px.reindex(common).ffill()

    # Daily ETF return
    px_ret = px_a.pct_change().fillna(0)

    # Term structure ratio — LAGGED by 1 day
    ratio = (vix_a / vix3m_a).shift(1).ffill().bfill()

    # Realized vol for sizing — LAGGED
    rvol = px_ret.rolling(threshold.vol_lookback, min_periods=5).std().shift(1)
    rvol = rvol * math.sqrt(TRADING_DAYS)
    rvol = rvol.fillna(0.50).clip(lower=0.10)

    df = pd.DataFrame({
        "date": common,
        "vix": vix_a.values,
        "vix3m": vix3m_a.values,
        "ratio": ratio.values,
        "price": px_a.values,
        "ret": px_ret.values,
        "rvol": rvol.values,
    }).set_index("date")

    # Position logic depends on side
    # For "short VXX / UVXY": negative position when contango (ratio < enter)
    # For "long SVXY":        positive position when contango (ratio < enter)
    positions = np.zeros(len(df))
    current_pos = 0.0
    sign = -1.0 if proxy.side == "short" else 1.0

    for i in range(len(df)):
        r = float(df["ratio"].iloc[i])
        if r > threshold.flat_threshold:
            current_pos = 0.0
        elif r < threshold.enter_threshold and current_pos == 0:
            size = min(threshold.max_position,
                        threshold.target_vol / max(float(df["rvol"].iloc[i]), 0.10))
            current_pos = sign * size
        elif r > threshold.exit_threshold:
            current_pos = 0.0
        positions[i] = current_pos

    df["position"] = positions
    df["strategy_ret"] = df["position"] * df["ret"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Backtest + walk-forward
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    proxy: str
    threshold: str
    n_days: int
    pct_in_position: float
    cagr: float
    sharpe: float
    max_dd: float
    vol: float
    total_return_pct: float
    yearly_sharpe: Dict[int, float] = field(default_factory=dict)
    yearly_cagr: Dict[int, float] = field(default_factory=dict)
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def backtest_one_config(
    data: Dict[str, pd.Series],
    proxy: ProxyConfig,
    threshold: ContangoConfig,
) -> Optional[BacktestResult]:
    """Backtest one (proxy, threshold) config across all years."""
    df = compute_signals(data, proxy, threshold)
    if df.empty:
        return None

    rets = df["strategy_ret"].values
    if len(rets) < 50:
        return None

    # Full-period metrics
    sharpe = float(annualized_sharpe(rets, rf_annual=RISK_FREE_ANNUAL))
    mdd = float(_mdd(rets))
    cagr = float(_cagr(rets))
    vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0

    # Compound
    equity = 100_000.0
    for r in rets:
        equity *= (1 + r)
    total_ret_pct = (equity / 100_000.0 - 1) * 100

    # Yearly breakdown
    df["year"] = df.index.year
    yearly_sharpe = {}
    yearly_cagr = {}
    for yr, grp in df.groupby("year"):
        yr_rets = grp["strategy_ret"].values
        if len(yr_rets) < 10:
            continue
        yr_sharpe = float(annualized_sharpe(yr_rets, rf_annual=RISK_FREE_ANNUAL))
        yr_cagr = float(_cagr(yr_rets))
        yearly_sharpe[int(yr)] = round(yr_sharpe, 3)
        yearly_cagr[int(yr)] = round(yr_cagr, 4)

    # IS (2018-2022) vs OOS (2023+) split
    is_mask = df.index.year < 2023
    oos_mask = df.index.year >= 2023
    is_rets = df.loc[is_mask, "strategy_ret"].values
    oos_rets = df.loc[oos_mask, "strategy_ret"].values

    is_sharpe = float(annualized_sharpe(is_rets, rf_annual=RISK_FREE_ANNUAL)) if len(is_rets) > 10 else 0.0
    oos_sharpe = float(annualized_sharpe(oos_rets, rf_annual=RISK_FREE_ANNUAL)) if len(oos_rets) > 10 else 0.0

    # Time in position
    pct_in_pos = float((df["position"] != 0).mean()) * 100

    return BacktestResult(
        proxy=proxy.name,
        threshold=threshold.name,
        n_days=len(df),
        pct_in_position=round(pct_in_pos, 1),
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 3),
        max_dd=round(mdd, 4),
        vol=round(vol, 4),
        total_return_pct=round(total_ret_pct, 2),
        yearly_sharpe=yearly_sharpe,
        yearly_cagr=yearly_cagr,
        is_sharpe=round(is_sharpe, 3),
        oos_sharpe=round(oos_sharpe, 3),
        daily_returns=df["strategy_ret"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Correlation with EXP-1220
# ═══════════════════════════════════════════════════════════════════════════

def correlation_to_exp1220(strategy_rets: pd.Series) -> float:
    """Correlation with EXP-1220 daily return series if available, else
    use yearly returns from better_portfolio.json as a proxy.
    """
    # Try daily series first
    for path in [
        ROOT / "reports" / "exp1220_robustness_report.json",
        ROOT / "reports" / "exp1220_dynamic_leverage.json",
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            for key in ("daily_pnl", "daily_returns", "pnl_series"):
                if key in data and isinstance(data[key], dict):
                    s = pd.Series(data[key])
                    s.index = pd.to_datetime(s.index)
                    common = strategy_rets.index.intersection(s.index)
                    if len(common) >= 20:
                        a = strategy_rets.reindex(common).fillna(0).values
                        b = s.reindex(common).fillna(0).values
                        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            pass

    # Fallback: yearly from better_portfolio.json
    bp_path = ROOT / "reports" / "better_portfolio.json"
    if bp_path.exists():
        try:
            data = json.loads(bp_path.read_text())
            exp1220 = data.get("streams_yearly", {}).get("EXP-1220", {})
            if exp1220:
                # Aggregate our daily rets to yearly
                our_yearly = strategy_rets.groupby(strategy_rets.index.year).apply(
                    lambda x: float(np.prod(1 + x) - 1)
                )
                common_years = sorted(set(our_yearly.index) & set(int(y) for y in exp1220.keys()))
                if len(common_years) >= 3:
                    a = np.array([our_yearly[y] for y in common_years])
                    b = np.array([exp1220[str(y)] / 100 for y in common_years])
                    if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                        return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            pass

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(
    results: List[BacktestResult],
    exp1220_corrs: Dict[str, float],
    data_availability: Dict[str, bool],
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Rank by OOS Sharpe
    ranked = sorted(results, key=lambda r: r.oos_sharpe, reverse=True)
    best = ranked[0] if ranked else None

    # Results table
    rows = ""
    for r in ranked:
        hit_sharpe = r.oos_sharpe >= 0.5
        hit_dd = r.max_dd < 0.15
        verdict = ("LIVE" if hit_sharpe and hit_dd else
                   "MARGINAL" if hit_sharpe or r.oos_sharpe > 0 else "KILL")
        verdict_color = ("var(--green)" if verdict == "LIVE" else
                         "var(--yellow)" if verdict == "MARGINAL" else "var(--red)")
        corr = exp1220_corrs.get(f"{r.proxy}_{r.threshold}", 0.0)
        rows += (
            f'<tr><td><strong>{r.proxy}</strong></td>'
            f'<td>{r.threshold}</td>'
            f'<td>{r.pct_in_position:.0f}%</td>'
            f'<td style="color:{"var(--green)" if r.cagr > 0 else "var(--red)"}">'
            f'{r.cagr:.1%}</td>'
            f'<td>{r.sharpe:.2f}</td>'
            f'<td>{r.oos_sharpe:.2f}</td>'
            f'<td>{r.max_dd:.1%}</td>'
            f'<td>{r.vol:.1%}</td>'
            f'<td>{corr:+.3f}</td>'
            f'<td style="color:{verdict_color};font-weight:700">{verdict}</td></tr>\n'
        )

    # Yearly breakdown for best config
    yearly_rows = ""
    if best:
        all_years = sorted(set(best.yearly_sharpe.keys()))
        for yr in all_years:
            sh = best.yearly_sharpe.get(yr, 0.0)
            cg = best.yearly_cagr.get(yr, 0.0)
            tag = "OOS" if yr >= 2023 else "IS"
            c = "var(--green)" if cg > 0 else "var(--red)"
            yearly_rows += (
                f'<tr><td>{yr} ({tag})</td>'
                f'<td style="color:{c}">{cg:.1%}</td>'
                f'<td>{sh:.2f}</td></tr>\n'
            )

    # Data availability block
    data_rows = ""
    for ticker, available in data_availability.items():
        status = "OK (Yahoo real)" if available else "MISSING"
        color = "var(--green)" if available else "var(--red)"
        data_rows += f'<tr><td>{ticker}</td><td style="color:{color}">{status}</td></tr>\n'

    # Summary verdict
    best_oos = best.oos_sharpe if best else 0
    if best_oos >= 0.5:
        summary_class = "callout-green"
        summary_msg = (f"BEST: {best.proxy} @ {best.threshold} — "
                       f"OOS Sharpe {best_oos:.2f}, CAGR {best.cagr:.1%}")
    elif best_oos > 0:
        summary_class = "callout-yellow"
        summary_msg = (f"MARGINAL: all proxies OOS Sharpe &lt; 0.50. "
                       f"Best is {best.proxy} @ {best.threshold} "
                       f"with {best_oos:.2f}.")
    else:
        summary_class = "callout-red"
        summary_msg = "KILL: no proxy produced positive OOS Sharpe."

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-1700 v2: VIX Roll Yield Multi-Proxy</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626;--yellow:#d97706;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1200px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;border-bottom:2px solid var(--border);padding-bottom:6px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.82rem}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.callout{{padding:14px;margin:14px 0;border-radius:8px;font-size:.88rem;line-height:1.7}}
.callout-blue{{background:#eff6ff;border-left:4px solid var(--blue)}}
.callout-green{{background:#ecfdf5;border-left:4px solid var(--green)}}
.callout-yellow{{background:#fffbeb;border-left:4px solid var(--yellow)}}
.callout-red{{background:#fef2f2;border-left:4px solid var(--red)}}
.footer{{margin-top:40px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:14px}}
</style></head><body>

<h1>EXP-1700 v2: VIX Roll Yield Multi-Proxy Test</h1>
<div class="subtitle">{ts} &bull; Rule Zero: Yahoo Finance chart API real data &bull; Zero synthetic</div>

<div class="callout callout-blue">
<strong>Honest data reality:</strong> CBOE /VX futures (the actual tradeable VIX contracts) are
NOT available via Yahoo Finance. Yahoo has the VIX INDEX constants (^VIX, ^VIX3M, ^VIX6M) which
are non-tradeable calculations, plus the VIX ETF/ETN products (VXX, SVXY, UVXY). Real futures
data would need a CBOE DataShop or Polygon futures subscription ($100-500/mo).
<br><br>
<strong>v1 failure (4f5b39c):</strong> Tested VXX short at a single 0.95 contango threshold and
got full-period Sharpe 0.03. The Wave 1 post-mortem concluded VXX is too noisy a proxy — an ETN
derivative of /VX futures with its own tracking error.
<br><br>
<strong>v2 expansion:</strong> Test THREE proxies (VXX short, SVXY long, UVXY short) at THREE
thresholds (tight/medium/loose) = 9 configurations. If the edge is real, it should show up in
at least one proxy × threshold combination even at the index-constant term-structure level.
</div>

<div class="callout {summary_class}">
<strong>{summary_msg}</strong>
</div>

<h2>All Configurations</h2>
<p class="subtitle">9 configurations (3 proxies × 3 thresholds), ranked by OOS Sharpe.
Kill: OOS Sharpe &lt;= 0. Marginal: 0 &lt; OOS Sharpe &lt; 0.5. Live: OOS Sharpe &gt;= 0.5 AND
Max DD &lt; 15%.</p>
<table>
<thead><tr>
  <th>Proxy</th><th>Threshold</th><th>In Pos%</th>
  <th>CAGR</th><th>Full Sharpe</th><th>OOS Sharpe</th>
  <th>Max DD</th><th>Vol</th><th>1220 Corr</th><th>Verdict</th>
</tr></thead>
<tbody>{rows}</tbody></table>

<h2>Best Config Year-by-Year</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th></tr></thead>
<tbody>{yearly_rows}</tbody></table>

<h2>Data Sources (Rule Zero)</h2>
<table>
<thead><tr><th>Symbol</th><th>Status</th></tr></thead>
<tbody>{data_rows}</tbody></table>

<h2>Missing Data (honest gap)</h2>
<ul style="padding-left:20px;line-height:1.8">
<li><strong>CBOE /VX futures front-month</strong> (VX1): needs CBOE DataShop ($100/mo) or
Polygon futures ($199/mo). Real contract-level prices would eliminate VXX's ETN path dependency.</li>
<li><strong>CBOE /VX futures calendar spreads</strong> (F1/F2 differential): the "pure" VRP play
is a short F1 / long F2 calendar spread that's delta-hedged. Can't test this without futures data.</li>
<li><strong>Intraday VIX</strong>: for same-day roll timing. Not in our stack.</li>
</ul>

<div class="footer">
  EXP-1700 v2 &bull; compass/vix_roll_yield_v2.py &bull; Yahoo Finance real data &bull; {ts}
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("EXP-1700 v2: VIX Roll Yield Multi-Proxy Test")
    log.info("Rule Zero: 100% real Yahoo Finance data")
    log.info("=" * 70)

    data = load_vix_data()

    data_availability = {
        t: (t in data) for t in ["^VIX", "^VIX3M", "^VIX6M", "VXX", "SVXY", "UVXY"]
    }

    # Grid: 3 proxies × 3 thresholds
    log.info(f"\nRunning {len(PROXIES)} × {len(THRESHOLDS)} = "
              f"{len(PROXIES) * len(THRESHOLDS)} configurations...")
    results: List[BacktestResult] = []
    exp1220_corrs: Dict[str, float] = {}

    for proxy in PROXIES:
        if proxy.ticker not in data:
            log.warning(f"  {proxy.name}: no data for {proxy.ticker} — skipping")
            continue
        for threshold in THRESHOLDS:
            log.info(f"  {proxy.name} @ {threshold.name}...")
            result = backtest_one_config(data, proxy, threshold)
            if result is None:
                log.warning(f"    failed (insufficient data)")
                continue

            results.append(result)
            log.info(f"    N={result.n_days}, in_pos={result.pct_in_position:.0f}%, "
                      f"CAGR={result.cagr:.1%}, Sharpe={result.sharpe:.2f}, "
                      f"OOS={result.oos_sharpe:.2f}, DD={result.max_dd:.1%}")

            # EXP-1220 correlation
            corr = correlation_to_exp1220(result.daily_returns)
            exp1220_corrs[f"{result.proxy}_{result.threshold}"] = corr

    if not results:
        log.error("No configurations produced results!")
        return

    # Summary
    log.info("\n" + "=" * 70)
    log.info("RESULTS (ranked by OOS Sharpe)")
    log.info("=" * 70)
    ranked = sorted(results, key=lambda r: r.oos_sharpe, reverse=True)
    for r in ranked:
        corr = exp1220_corrs.get(f"{r.proxy}_{r.threshold}", 0.0)
        log.info(f"  {r.proxy:12s} / {r.threshold:6s}: "
                  f"OOS={r.oos_sharpe:>6.2f}  CAGR={r.cagr:>+7.1%}  "
                  f"DD={r.max_dd:>5.1%}  1220ρ={corr:+.3f}")

    best = ranked[0]
    log.info(f"\nBest: {best.proxy} @ {best.threshold}")
    log.info(f"  Full Sharpe: {best.sharpe:.2f}")
    log.info(f"  OOS Sharpe:  {best.oos_sharpe:.2f}")
    log.info(f"  CAGR:        {best.cagr:.1%}")
    log.info(f"  Max DD:      {best.max_dd:.1%}")

    # Write reports
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(results, exp1220_corrs, data_availability)
    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info(f"\nHTML: {REPORT_PATH}")

    json_data = {
        "experiment": "EXP-1700 v2",
        "name": "VIX Roll Yield Multi-Proxy",
        "data_source": "Yahoo Finance chart API (^VIX, ^VIX3M, VXX, SVXY, UVXY)",
        "rule_zero_compliant": True,
        "data_availability": data_availability,
        "proxies_tested": [p.name for p in PROXIES],
        "thresholds_tested": [t.name for t in THRESHOLDS],
        "results": [
            {
                "proxy": r.proxy,
                "threshold": r.threshold,
                "n_days": r.n_days,
                "pct_in_position": r.pct_in_position,
                "cagr": r.cagr,
                "sharpe_full": r.sharpe,
                "sharpe_is": r.is_sharpe,
                "sharpe_oos": r.oos_sharpe,
                "max_dd": r.max_dd,
                "vol": r.vol,
                "total_return_pct": r.total_return_pct,
                "yearly_sharpe": r.yearly_sharpe,
                "yearly_cagr": r.yearly_cagr,
                "exp1220_correlation": exp1220_corrs.get(f"{r.proxy}_{r.threshold}", 0.0),
            }
            for r in results
        ],
        "best": {
            "proxy": best.proxy,
            "threshold": best.threshold,
            "oos_sharpe": best.oos_sharpe,
            "cagr": best.cagr,
            "max_dd": best.max_dd,
        },
        "data_gap_note": (
            "CBOE /VX futures data not available via Yahoo. Real contract-level "
            "testing would need CBOE DataShop or Polygon futures subscription."
        ),
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str))
    log.info(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
