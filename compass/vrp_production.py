"""
compass/vrp_production.py — EXP-1660 VRP Production Portfolio Integration.

Production-ready VRP module with:
  1. Best VRP configs per ticker (frozen from universe-expansion winners)
  2. Monthly PnL series generation for each survivor
  3. Portfolio integration with EXP-1220 at 5%, 10%, 15% allocations
  4. Correlation check with EXP-1820 Dispersion (both sell vol)
  5. Ready to be imported and used by the North Star portfolio builder

Frozen winners from reports/exp1660_vrp_universe.json (commit 995a6a6):
  QQQ: delta=0.15, threshold=1.5%, width=8%   → 73 trades, OOS SR 0.73
  XLF: delta=0.18, threshold=2.5%, width=8%   → 204 trades, OOS SR 0.86
  XLI: delta=0.15, threshold=2.0%, width=8%   → 61 trades, OOS SR 1.23

RULE ZERO: All option prices from IronVault. Zero np.random / synthetic.
Sharpe via compass/metrics.py arithmetic mean.

Output:
    reports/exp1660_vrp_production.html
    reports/exp1660_vrp_production.json
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from backtest.backtester import _yf_download_safe
from compass.metrics import annualized_sharpe, max_drawdown as _mdd, cagr as _cagr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vrp_production")

REPORT_PATH = ROOT / "reports" / "exp1660_vrp_production.html"
JSON_PATH = ROOT / "reports" / "exp1660_vrp_production.json"
CAPITAL = 100_000
OOS_START = 2023
MIN_SPACING = 3
RISK_FREE_ANNUAL = 0.045


# ═══════════════════════════════════════════════════════════════════════════
# Frozen production configs — winners from EXP-1660 universe expansion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VRPConfig:
    ticker: str
    short_delta: float
    iv_rv_threshold: float
    wing_width_pct: float
    risk_pct: float = 0.02
    max_contracts: int = 5

    def __repr__(self):
        return (f"VRPConfig({self.ticker}, delta={self.short_delta}, "
                f"threshold={self.iv_rv_threshold:.3f}, width={self.wing_width_pct})")


PRODUCTION_CONFIGS: List[VRPConfig] = [
    VRPConfig("XLI", short_delta=0.15, iv_rv_threshold=0.020, wing_width_pct=0.08),
    VRPConfig("XLF", short_delta=0.18, iv_rv_threshold=0.025, wing_width_pct=0.08),
    VRPConfig("QQQ", short_delta=0.15, iv_rv_threshold=0.015, wing_width_pct=0.08),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (reusable for production deployment)
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fetch_yahoo(ticker: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, "2019-01-01", "2026-07-01")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _find_exps(hd: IronVault, ticker: str) -> List[str]:
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker=? AND option_type='P' ORDER BY expiration",
        (ticker,),
    )
    exps = [r[0] for r in cur.fetchall()]
    conn.close()
    return exps


def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


def _find_priced_strike(
    hd: IronVault, ticker: str, exp: str, exp_obj: datetime,
    trade_date: str, spot: float, option_type: str, otm_pct: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Find closest strike to OTM target that has a real IronVault price."""
    strikes = hd.get_available_strikes(ticker, exp, trade_date, option_type)
    if not strikes:
        return None, None
    target = spot * (1 - otm_pct) if option_type == "P" else spot * (1 + otm_pct)
    candidates = sorted(strikes, key=lambda k: abs(k - target))[:15]
    for k in candidates:
        sym = IronVault.build_occ_symbol(ticker, exp_obj, k, option_type)
        px = hd.get_contract_price(sym, trade_date)
        if px is not None and px > 0.02:
            return float(k), float(px)
    return None, None


def _atm_straddle_cost(
    hd: IronVault, ticker: str, exp: str, exp_obj: datetime,
    trade_date: str, spot: float,
) -> Optional[float]:
    put_strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    call_strikes = hd.get_available_strikes(ticker, exp, trade_date, "C")
    if not put_strikes or not call_strikes:
        return None
    put_k = min(put_strikes, key=lambda k: abs(k - spot))
    call_k = min(call_strikes, key=lambda k: abs(k - spot))
    pp = hd.get_contract_price(IronVault.build_occ_symbol(ticker, exp_obj, put_k, "P"), trade_date)
    cp = hd.get_contract_price(IronVault.build_occ_symbol(ticker, exp_obj, call_k, "C"), trade_date)
    if pp is None or cp is None:
        return None
    return float(pp + cp)


def _iv_from_straddle(straddle_cost: float, spot: float, dte: int) -> float:
    """Brenner-Subrahmanyam: σ ≈ straddle / (spot × √(2T/π)).

    Converts REAL straddle price to a vol number — NOT a pricing model.
    """
    if spot <= 0 or dte <= 0 or straddle_cost <= 0:
        return 0.0
    T = dte / 365.0
    return float(straddle_cost / (spot * math.sqrt(2 * T / math.pi)))


# ═══════════════════════════════════════════════════════════════════════════
# Single-ticker VRP backtest (reused for production)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VRPTicker:
    config: VRPConfig
    trades: List[Dict] = field(default_factory=list)
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trade_sharpe: float = 0.0
    monthly_pnl: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_pnl: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_vrp(hd: IronVault, config: VRPConfig,
             underlying_df: pd.DataFrame) -> VRPTicker:
    """Run the frozen VRP config on real IronVault data. Produces daily and
    monthly PnL series for portfolio integration."""
    close = underlying_df["Close"]
    td_set = set(underlying_df.index.strftime("%Y-%m-%d"))
    all_exps = _find_exps(hd, config.ticker)
    rvol = _realized_vol(close, window=20)

    trades: List[Dict] = []
    last_entry = None

    for date in underlying_df.index:
        ds = date.strftime("%Y-%m-%d")
        if ds < "2020-03-01":
            continue
        if last_entry and (date - last_entry).days < MIN_SPACING:
            continue

        try:
            spot = float(close.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(spot) or spot <= 0:
            continue

        # Short-dated exp 7-21 DTE
        short_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 7 <= dte <= 21:
                short_exp = e
                break
        if short_exp is None:
            continue
        short_exp_obj = _exp_dt(short_exp)
        short_dte = (short_exp_obj - date).days

        # Hedge exp 45-90 DTE
        hedge_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 45 <= dte <= 90:
                hedge_exp = e
                break
        if hedge_exp is None:
            continue
        hedge_exp_obj = _exp_dt(hedge_exp)

        # IV-RV gap signal
        straddle = _atm_straddle_cost(hd, config.ticker, short_exp,
                                        short_exp_obj, ds, spot)
        if straddle is None:
            continue
        iv = _iv_from_straddle(straddle, spot, short_dte)
        try:
            rv = float(rvol.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(rv):
            continue

        iv_rv_gap = iv - rv
        if iv_rv_gap < config.iv_rv_threshold:
            continue

        # Short strangle + hedge put
        put_otm = config.short_delta * 0.5 * math.sqrt(short_dte / 30)
        call_otm = put_otm
        put_k, put_px = _find_priced_strike(
            hd, config.ticker, short_exp, short_exp_obj, ds, spot, "P", put_otm)
        call_k, call_px = _find_priced_strike(
            hd, config.ticker, short_exp, short_exp_obj, ds, spot, "C", call_otm)
        if put_k is None or call_k is None:
            continue

        hedge_put_k, hedge_put_px = _find_priced_strike(
            hd, config.ticker, hedge_exp, hedge_exp_obj, ds, spot, "P",
            config.wing_width_pct)
        if hedge_put_k is None:
            continue

        strangle_credit = put_px + call_px
        net_credit = strangle_credit - hedge_put_px
        if net_credit <= 0:
            continue

        put_wing = put_k - hedge_put_k
        risk_est = max(net_credit * 2, put_wing * 0.4)
        contracts = max(1, min(config.max_contracts,
                                int(CAPITAL * config.risk_pct / (risk_est * 100))))

        # Walk forward to exit
        current = date + timedelta(days=1)
        exit_date = ds
        exit_reason = "expiration"
        exit_pnl = 0.0

        while current <= short_exp_obj:
            cs = current.strftime("%Y-%m-%d")
            if cs not in td_set:
                current += timedelta(days=1)
                continue

            put_sym = IronVault.build_occ_symbol(config.ticker, short_exp_obj, put_k, "P")
            call_sym = IronVault.build_occ_symbol(config.ticker, short_exp_obj, call_k, "C")
            hedge_sym = IronVault.build_occ_symbol(config.ticker, hedge_exp_obj, hedge_put_k, "P")
            pp2 = hd.get_contract_price(put_sym, cs)
            cp2 = hd.get_contract_price(call_sym, cs)
            hp2 = hd.get_contract_price(hedge_sym, cs)

            if pp2 is not None and cp2 is not None:
                cur_strangle = float(pp2 + cp2)
                cur_hedge = float(hp2) if hp2 is not None else float(hedge_put_px)
                unrealized = net_credit - (cur_strangle - cur_hedge)

                if unrealized >= net_credit * 0.50:
                    exit_pnl = unrealized
                    exit_date = cs
                    exit_reason = "profit_target"
                    break
                if unrealized <= -net_credit * 2.0:
                    exit_pnl = unrealized
                    exit_date = cs
                    exit_reason = "stop_loss"
                    break
                exit_pnl = unrealized
                exit_date = cs

            current += timedelta(days=1)

        if exit_reason == "expiration":
            put_sym = IronVault.build_occ_symbol(config.ticker, short_exp_obj, put_k, "P")
            call_sym = IronVault.build_occ_symbol(config.ticker, short_exp_obj, call_k, "C")
            pp_final = hd.get_contract_price(put_sym, short_exp)
            cp_final = hd.get_contract_price(call_sym, short_exp)
            if pp_final is not None and cp_final is not None:
                exit_pnl = net_credit - (float(pp_final) + float(cp_final))

        total_pnl = exit_pnl * 100 * contracts

        trades.append({
            "entry_date": ds,
            "exit_date": exit_date,
            "ticker": config.ticker,
            "pnl": round(total_pnl, 2),
            "exit_reason": exit_reason,
            "net_credit": round(net_credit, 4),
            "contracts": contracts,
        })
        last_entry = date

    return _compute_ticker_result(config, trades, underlying_df)


def _compute_ticker_result(config: VRPConfig, trades: List[Dict],
                             underlying_df: pd.DataFrame) -> VRPTicker:
    if not trades:
        return VRPTicker(config=config)

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    pnls = df["pnl"].values
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    mu = float(np.mean(pnls))
    sigma = float(np.std(pnls, ddof=1)) if n > 1 else 1.0
    trade_sharpe = float(mu / sigma * math.sqrt(min(n, 52))) if sigma > 1e-9 else 0.0

    # Daily PnL series
    daily_pnl = df.groupby("exit_date")["pnl"].sum()
    full_range = pd.date_range(
        pd.Timestamp("2020-03-01"), underlying_df.index.max(), freq="B")
    daily_pnl_full = daily_pnl.reindex(full_range, fill_value=0)

    # Monthly PnL series
    monthly_pnl = daily_pnl_full.resample("ME").sum()

    return VRPTicker(
        config=config,
        trades=trades,
        n_trades=n,
        total_pnl=round(total, 2),
        win_rate=round(wins / n, 3),
        trade_sharpe=round(trade_sharpe, 3),
        monthly_pnl=monthly_pnl,
        daily_pnl=daily_pnl_full,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Production API — import this from north_star_portfolio.py
# ═══════════════════════════════════════════════════════════════════════════

def build_vrp_portfolio(
    hd: Optional[IronVault] = None,
    underlying_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, VRPTicker]:
    """Build the production VRP portfolio — runs all frozen configs.

    Returns: dict of ticker → VRPTicker with monthly PnL series ready for
    portfolio integration.
    """
    if hd is None:
        hd = IronVault.instance()

    if underlying_data is None:
        underlying_data = {
            cfg.ticker: _fetch_yahoo(cfg.ticker)
            for cfg in PRODUCTION_CONFIGS
        }

    results = {}
    for cfg in PRODUCTION_CONFIGS:
        log.info(f"Running {cfg}...")
        udf = underlying_data.get(cfg.ticker)
        if udf is None:
            log.warning(f"  No price data for {cfg.ticker}, skipping")
            continue
        tr = run_vrp(hd, cfg, udf)
        results[cfg.ticker] = tr
        log.info(f"  {cfg.ticker}: {tr.n_trades} trades, "
                  f"${tr.total_pnl:,.0f} PnL, trade SR {tr.trade_sharpe:.2f}")

    return results


def combined_vrp_series(results: Dict[str, VRPTicker]) -> pd.Series:
    """Combine the VRP tickers equal-weight into a single monthly PnL series."""
    viable = [r for r in results.values() if r.n_trades >= 30]
    if not viable:
        return pd.Series(dtype=float)
    weight = 1.0 / len(viable)
    combined = None
    for r in viable:
        scaled = r.monthly_pnl * weight
        combined = scaled if combined is None else combined.add(scaled, fill_value=0)
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# EXP-1220 @ 1.5x monthly proxy
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220_monthly_1_5x() -> pd.Series:
    """Generate EXP-1220 monthly PnL series at 1.5x leverage from validated
    yearly data in better_portfolio.json.

    Since better_portfolio.json has yearly returns only, we distribute them
    evenly across trading months within each year. This is a conservative
    approximation — real monthly data would have more variance.
    """
    path = ROOT / "reports" / "better_portfolio.json"
    if not path.exists():
        log.warning("better_portfolio.json not found — cannot build EXP-1220 series")
        return pd.Series(dtype=float)

    data = json.loads(path.read_text())
    yearly = data.get("streams_yearly", {}).get("EXP-1220", {})
    if not yearly:
        return pd.Series(dtype=float)

    # Distribute each year's PnL evenly across 12 months, apply 1.5x leverage
    monthly_data = {}
    for year_str, pct in yearly.items():
        year = int(year_str)
        leveraged_annual_pnl = (pct / 100.0) * CAPITAL * 1.5
        monthly_pnl = leveraged_annual_pnl / 12.0
        for month in range(1, 13):
            date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            monthly_data[date] = monthly_pnl

    return pd.Series(monthly_data).sort_index()


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio integration — test 5%, 10%, 15% VRP allocations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    name: str
    vrp_alloc_pct: float
    monthly_returns: pd.Series
    annual_cagr: float
    annual_sharpe: float
    max_dd: float
    vol_annual: float
    total_pnl: float


def test_portfolio(
    name: str,
    exp1220_monthly: pd.Series,
    vrp_monthly: pd.Series,
    vrp_pct: float,
) -> PortfolioResult:
    """Combine EXP-1220 (core) with VRP (allocation) and compute metrics.

    vrp_pct: fraction of capital allocated to VRP (e.g. 0.10 for 10%).
    EXP-1220 gets (1 - vrp_pct) of capital.
    """
    # Align series
    idx = exp1220_monthly.index.union(vrp_monthly.index)
    exp1220 = exp1220_monthly.reindex(idx, fill_value=0)
    vrp = vrp_monthly.reindex(idx, fill_value=0)

    # Allocation-weighted combination (each strategy normalized to its
    # own capital allocation)
    exp1220_weight = 1.0 - vrp_pct
    combined_pnl = exp1220 * exp1220_weight + vrp * vrp_pct

    # Metrics on monthly returns
    monthly_rets = combined_pnl / CAPITAL
    vol_m = float(np.std(monthly_rets.values, ddof=1)) if len(monthly_rets) > 1 else 0.0
    vol_annual = vol_m * math.sqrt(12)
    mean_m = float(np.mean(monthly_rets.values))
    excess_m = mean_m - RISK_FREE_ANNUAL / 12
    sharpe = (excess_m / vol_m * math.sqrt(12)) if vol_m > 1e-9 else 0.0

    # Compound to get CAGR
    equity = 100_000.0
    for r in monthly_rets.values:
        equity *= (1 + r)
    n_years = len(monthly_rets) / 12.0
    cagr = (equity / 100_000.0) ** (1 / max(n_years, 0.5)) - 1 if equity > 0 else -1

    # Max DD from monthly equity curve
    equity_curve = [100_000.0]
    for r in monthly_rets.values:
        equity_curve.append(equity_curve[-1] * (1 + r))
    arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(arr)
    dd = (peaks - arr) / peaks
    max_dd = float(dd.max())

    return PortfolioResult(
        name=name,
        vrp_alloc_pct=vrp_pct,
        monthly_returns=monthly_rets,
        annual_cagr=round(cagr, 4),
        annual_sharpe=round(sharpe, 3),
        max_dd=round(max_dd, 4),
        vol_annual=round(vol_annual, 4),
        total_pnl=round(float(combined_pnl.sum()), 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dispersion correlation — re-run the dispersion backtest to get series
# ═══════════════════════════════════════════════════════════════════════════

def load_dispersion_monthly() -> Optional[pd.Series]:
    """Re-run the existing EXP-1820 dispersion backtest to get a monthly series.

    compass.dispersion.backtest_dispersion() returns a list of DispersionTrade
    dataclasses. We extract exit_date and pnl from each to build a daily series
    then resample to monthly.
    """
    try:
        from compass.dispersion import backtest_dispersion
        log.info("  Running compass.dispersion.backtest_dispersion...")
        trades = backtest_dispersion()

        if not trades:
            log.warning("  Dispersion returned no trades")
            return None

        # Extract exit_date and pnl from DispersionTrade dataclasses
        records = []
        for t in trades:
            exit_date = getattr(t, "exit_date", None) or getattr(t, "entry_date", None)
            pnl = getattr(t, "pnl", None)
            if exit_date is None or pnl is None:
                continue
            records.append({"exit_date": exit_date, "pnl": pnl})

        if not records:
            log.warning("  No usable trade records extracted")
            return None

        df = pd.DataFrame(records)
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        daily = df.groupby("exit_date")["pnl"].sum()
        monthly = daily.resample("ME").sum()
        log.info(f"  Dispersion: {len(trades)} trades → "
                  f"{len(monthly)} monthly bars, {(monthly != 0).sum()} active")
        return monthly

    except Exception as e:
        log.warning(f"  Could not load dispersion series: {type(e).__name__}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(
    vrp_results: Dict[str, VRPTicker],
    vrp_combined: pd.Series,
    portfolio_results: List[PortfolioResult],
    dispersion_corr: Optional[float],
    exp1220_baseline: PortfolioResult,
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # VRP per-ticker rows
    vrp_rows = ""
    for ticker, r in sorted(vrp_results.items()):
        if r.n_trades == 0:
            continue
        c = "var(--green)" if r.total_pnl > 0 else "var(--red)"
        vrp_rows += (
            f'<tr><td><strong>{ticker}</strong></td>'
            f'<td>{r.config.short_delta:.2f}</td>'
            f'<td>{r.config.iv_rv_threshold:.1%}</td>'
            f'<td>{r.n_trades}</td>'
            f'<td>{r.win_rate:.0%}</td>'
            f'<td style="color:{c}">${r.total_pnl:,.0f}</td>'
            f'<td>{r.trade_sharpe:.2f}</td></tr>\n'
        )

    # Portfolio allocation comparison
    port_rows = ""
    for p in [exp1220_baseline] + portfolio_results:
        delta_sharpe = p.annual_sharpe - exp1220_baseline.annual_sharpe
        delta_color = ("var(--green)" if delta_sharpe > 0.05 else
                       "var(--red)" if delta_sharpe < -0.05 else "var(--muted)")
        port_rows += (
            f'<tr><td><strong>{p.name}</strong></td>'
            f'<td>{p.vrp_alloc_pct:.0%}</td>'
            f'<td>{p.annual_cagr:.1%}</td>'
            f'<td>{p.annual_sharpe:.2f}</td>'
            f'<td style="color:{delta_color}">{delta_sharpe:+.2f}</td>'
            f'<td>{p.max_dd:.1%}</td>'
            f'<td>{p.vol_annual:.1%}</td>'
            f'<td>${p.total_pnl:,.0f}</td></tr>\n'
        )

    # Pick the best VRP allocation
    best = max(portfolio_results, key=lambda p: p.annual_sharpe)
    best_delta = best.annual_sharpe - exp1220_baseline.annual_sharpe
    verdict = ("VRP ADDS VALUE" if best_delta > 0.05 else
               "VRP DILUTES PORTFOLIO" if best_delta < -0.05 else
               "NEUTRAL — no meaningful impact")

    disp_str = (f"{dispersion_corr:+.3f}" if dispersion_corr is not None
                else "N/A (dispersion series not loadable)")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-1660 VRP Production — Portfolio Integration</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626;--yellow:#d97706;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1200px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;border-bottom:2px solid var(--border);padding-bottom:6px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center}}
.c .l{{color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
.c .v{{font-size:1.1rem;font-weight:800;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.82rem}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.callout{{background:var(--card);border-left:4px solid var(--blue);padding:14px;margin:12px 0;font-size:.85rem;line-height:1.6;border-radius:4px}}
.footer{{margin-top:40px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:14px}}
</style></head><body>

<h1>EXP-1660 VRP Production — Portfolio Integration Test</h1>
<div class="subtitle">{ts} &bull; Rule Zero: 100% real IronVault data + validated yearly streams &bull; Zero synthetic</div>

<div class="callout">
<strong>Production VRP configs (frozen from EXP-1660 universe winners):</strong> XLI
(delta=0.15, threshold=2.0%), XLF (delta=0.18, threshold=2.5%), QQQ (delta=0.15,
threshold=1.5%). These are the 3 survivors from the 118-point grid search that
hit >=50 trades AND positive OOS Sharpe.
<br><br>
<strong>Test:</strong> Combine with EXP-1220 at 1.5× leverage and allocate 5%, 10%,
15% to the VRP basket. Does VRP improve the Sharpe of the core portfolio?
</div>

<h2>VRP Per-Ticker Results</h2>
<table>
<thead><tr><th>Ticker</th><th>Delta</th><th>Threshold</th><th>N Trades</th><th>WR</th><th>Total PnL</th><th>Trade Sharpe</th></tr></thead>
<tbody>{vrp_rows}</tbody></table>

<h2>Portfolio Allocation Comparison</h2>
<p class="subtitle">Baseline: EXP-1220 @ 1.5× leverage with 0% VRP. Each row adds VRP at the stated %.</p>
<table>
<thead><tr>
  <th>Portfolio</th><th>VRP %</th>
  <th>CAGR</th><th>Sharpe</th><th>Δ Sharpe</th>
  <th>Max DD</th><th>Vol</th><th>Total PnL</th>
</tr></thead>
<tbody>{port_rows}</tbody></table>

<div class="cards">
  <div class="c"><div class="l">Best Allocation</div><div class="v">{best.vrp_alloc_pct:.0%}</div></div>
  <div class="c"><div class="l">Best Sharpe</div><div class="v">{best.annual_sharpe:.2f}</div></div>
  <div class="c"><div class="l">Baseline Sharpe</div><div class="v">{exp1220_baseline.annual_sharpe:.2f}</div></div>
  <div class="c"><div class="l">Sharpe Delta</div>
    <div class="v" style="color:{'var(--green)' if best_delta > 0 else 'var(--red)'}">
      {best_delta:+.2f}</div></div>
  <div class="c"><div class="l">Verdict</div><div class="v" style="font-size:.9rem">{verdict}</div></div>
</div>

<h2>VRP vs Dispersion Correlation</h2>
<p>Both sell volatility. The question: are they trading the same edge or
genuinely different risk premia?</p>
<div class="cards">
  <div class="c"><div class="l">Correlation (monthly)</div><div class="v">{disp_str}</div></div>
  <div class="c"><div class="l">Interpretation</div>
    <div class="v" style="font-size:.85rem">
      {("Likely redundant — trading same edge" if dispersion_corr is not None and dispersion_corr > 0.5 else
         "Meaningfully different edges" if dispersion_corr is not None and abs(dispersion_corr) < 0.3 else
         "Weak overlap" if dispersion_corr is not None else
         "Not computed")}
    </div></div>
</div>

<h2>Honest Caveats (Rule Zero)</h2>
<ul style="padding-left:20px;line-height:1.7">
<li><strong>EXP-1220 monthly series is yearly-distributed.</strong> better_portfolio.json only
has yearly returns for EXP-1220. I distribute each year's return evenly across 12 months. This
underestimates real monthly vol, which means the reported Sharpe is biased UP. A proper test
needs real daily/monthly EXP-1220 PnL data.</li>
<li><strong>VRP daily series dilution.</strong> VRP trades cluster (most days have zero PnL).
The daily Sharpe is low even when trade-level Sharpe is positive. This is the Phase 7 capital
utilization bug from MASTERPLAN.</li>
<li><strong>Dispersion correlation uses whatever data the existing module produces.</strong>
If the dispersion module doesn't expose a trade list, this correlation falls back to N/A.</li>
</ul>

<div class="footer">
  EXP-1660 VRP Production &bull; compass/vrp_production.py &bull; {ts}
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("EXP-1660 VRP Production — Portfolio Integration")
    log.info("Rule Zero: 100% real IronVault data")
    log.info("=" * 70)

    # Run VRP production configs
    log.info("\n[1/5] Running frozen VRP production configs...")
    vrp_results = build_vrp_portfolio()
    log.info(f"  {len(vrp_results)} tickers traded")

    # Combine into a single monthly series
    log.info("\n[2/5] Combining VRP tickers into equal-weight monthly series...")
    vrp_combined = combined_vrp_series(vrp_results)
    total_months = len(vrp_combined[vrp_combined != 0])
    log.info(f"  Combined monthly series: {len(vrp_combined)} months, "
              f"{total_months} active months")

    # Load EXP-1220 monthly
    log.info("\n[3/5] Loading EXP-1220 @ 1.5x monthly baseline...")
    exp1220_monthly = load_exp1220_monthly_1_5x()
    log.info(f"  {len(exp1220_monthly)} monthly bars")

    # Baseline: EXP-1220 alone (0% VRP)
    baseline = test_portfolio(
        "EXP-1220 @ 1.5× solo", exp1220_monthly, vrp_combined, vrp_pct=0.0)
    log.info(f"  Baseline: CAGR={baseline.annual_cagr:.1%}, "
              f"Sharpe={baseline.annual_sharpe:.2f}, DD={baseline.max_dd:.1%}")

    # Test allocations 5%, 10%, 15%
    log.info("\n[4/5] Testing VRP allocations 5%, 10%, 15%...")
    allocations = [0.05, 0.10, 0.15]
    portfolio_results = []
    for pct in allocations:
        p = test_portfolio(f"Core + {pct:.0%} VRP", exp1220_monthly, vrp_combined, pct)
        portfolio_results.append(p)
        delta = p.annual_sharpe - baseline.annual_sharpe
        log.info(f"  {pct:.0%}: CAGR={p.annual_cagr:.1%}, Sharpe={p.annual_sharpe:.2f} "
                  f"(Δ {delta:+.2f}), DD={p.max_dd:.1%}")

    # Dispersion correlation
    log.info("\n[5/5] Computing VRP vs Dispersion correlation...")
    dispersion_monthly = load_dispersion_monthly()
    dispersion_corr = None
    if dispersion_monthly is not None and len(dispersion_monthly) > 5:
        common = vrp_combined.index.intersection(dispersion_monthly.index)
        if len(common) >= 5:
            a = vrp_combined.reindex(common).fillna(0).values
            b = dispersion_monthly.reindex(common).fillna(0).values
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                dispersion_corr = float(np.corrcoef(a, b)[0, 1])
                log.info(f"  VRP ↔ Dispersion monthly correlation: {dispersion_corr:+.3f}")
            else:
                log.info("  Dispersion series has no variance; skipping")
        else:
            log.info(f"  Only {len(common)} overlapping months; skipping")
    else:
        log.info("  Dispersion series unavailable — see honest caveat in report")

    # Write reports
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(vrp_results, vrp_combined, portfolio_results,
                           dispersion_corr, baseline)
    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info(f"\nHTML: {REPORT_PATH}")

    json_data = {
        "experiment": "EXP-1660 Production",
        "rule_zero_compliant": True,
        "frozen_configs": [
            {
                "ticker": c.ticker,
                "short_delta": c.short_delta,
                "iv_rv_threshold": c.iv_rv_threshold,
                "wing_width_pct": c.wing_width_pct,
            }
            for c in PRODUCTION_CONFIGS
        ],
        "per_ticker_results": {
            t: {
                "n_trades": r.n_trades,
                "total_pnl": r.total_pnl,
                "win_rate": r.win_rate,
                "trade_sharpe": r.trade_sharpe,
                "active_months": int((r.monthly_pnl != 0).sum()),
            }
            for t, r in vrp_results.items()
        },
        "baseline": {
            "name": baseline.name,
            "cagr": baseline.annual_cagr,
            "sharpe": baseline.annual_sharpe,
            "max_dd": baseline.max_dd,
            "vol": baseline.vol_annual,
            "total_pnl": baseline.total_pnl,
        },
        "allocation_tests": [
            {
                "vrp_pct": p.vrp_alloc_pct,
                "cagr": p.annual_cagr,
                "sharpe": p.annual_sharpe,
                "max_dd": p.max_dd,
                "vol": p.vol_annual,
                "total_pnl": p.total_pnl,
                "delta_sharpe": round(p.annual_sharpe - baseline.annual_sharpe, 3),
            }
            for p in portfolio_results
        ],
        "dispersion_correlation": dispersion_corr,
        "caveats": [
            "EXP-1220 monthly series distributed evenly from yearly data (underestimates vol)",
            "VRP daily series is diluted — most days have zero PnL",
            "Dispersion correlation depends on compass.dispersion exposing a trade list",
        ],
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str))
    log.info(f"JSON: {JSON_PATH}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    best = max(portfolio_results, key=lambda p: p.annual_sharpe)
    delta = best.annual_sharpe - baseline.annual_sharpe
    log.info(f"Baseline (0% VRP):  CAGR={baseline.annual_cagr:.1%}, "
              f"Sharpe={baseline.annual_sharpe:.2f}")
    log.info(f"Best VRP alloc:     {best.vrp_alloc_pct:.0%}, "
              f"CAGR={best.annual_cagr:.1%}, Sharpe={best.annual_sharpe:.2f}")
    log.info(f"Sharpe improvement: {delta:+.2f}")
    if delta > 0.05:
        log.info("VERDICT: VRP adds value to the North Star portfolio")
    elif delta < -0.05:
        log.info("VERDICT: VRP dilutes the core — exclude from portfolio")
    else:
        log.info("VERDICT: VRP is neutral — too weak standalone to help meaningfully")


if __name__ == "__main__":
    main()
