"""
EXP-1840: IV Spike Entry — Volatility-Timed Credit Spread Overlay
==================================================================

Thesis
------
Credit spread sellers are paid to absorb volatility risk. The more IV
spikes above recent norms, the more premium we collect. Instead of
entering on a fixed cadence (EXP-1220's 7d), time entries to moments
when intraday IV spikes above its 5-day rolling mean by > 1.5 σ.

Signal
------
For each day, compute an ATM IV proxy at end-of-day close:
  - SPY close price → nearest 0.5% OTM strike
  - Front-month (20-45 DTE) ATM put/call pair
  - IV proxy = (atm_put_price + atm_call_price) / SPY  (straddle proxy)

Rolling 5-day mean and std of the IV proxy. A spike is defined as:
  spike_z = (today - 5d_mean) / 5d_std >= 1.5

On spike days, enter a bull put spread at close. Exit at 50% profit,
2× stop loss, or 5 DTE, whichever comes first.

Data
----
REAL IronVault data only:
  - option_daily for ATM straddle price proxy
  - option_contracts for strike discovery
  - get_spread_prices for entry/exit pricing

ZERO SYNTHETIC DATA. Zero np.random.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault

logger = logging.getLogger(__name__)

# ─── Strategy parameters ────────────────────────────────────────────────────
TICKER = "SPY"
CAPITAL = 100_000.0

# IV spike detection
IV_LOOKBACK = 5            # 5-day rolling window
IV_SPIKE_Z = 1.5           # enter when z >= 1.5
IV_MIN_LEVEL = 0.02        # min absolute IV proxy to avoid noise

# Spread construction
TARGET_DTE_MIN = 20        # 20-45 DTE
TARGET_DTE_MAX = 45
TARGET_DTE_PREFERRED = 30
SPREAD_WIDTH = 5.0         # $5 wide
OTM_PCT = 0.03             # short strike 3% OTM (closer = more premium on spike)
MIN_CREDIT_PCT = 0.15      # minimum 15% of width (i.e. $0.75 credit on $5 wide)

# Exit rules
PROFIT_TARGET_PCT = 0.50   # close at 50% of max profit
STOP_LOSS_MULT = 2.0       # stop when loss = 2× credit
MIN_DTE_EXIT = 5           # close if DTE <= 5
MAX_HOLD_DAYS = 20

# Sizing
RISK_PCT = 0.02            # 2% per trade
MIN_SPACING_DAYS = 3       # min days between entries

# Walk-forward
IS_END_YEAR = 2022
OOS_START_YEAR = 2023
TRADING_DAYS = 252

# Real-data window
START_DATE = "2020-01-02"
END_DATE = "2025-12-31"


# ─── Data classes ──────────────────────────────────────────────────────────
@dataclass
class IVObservation:
    date: str
    spy_price: float
    atm_strike: float
    expiration: str
    put_price: float
    call_price: float
    iv_proxy: float        # (put + call) / spy
    iv_ma: Optional[float] = None
    iv_std: Optional[float] = None
    iv_z: Optional[float] = None
    is_spike: bool = False


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    expiration: str
    short_strike: float
    long_strike: float
    entry_credit: float
    exit_debit: float
    contracts: int
    pnl: float
    hold_days: int
    exit_reason: str
    iv_z_at_entry: float
    iv_proxy_at_entry: float


@dataclass
class BacktestResult:
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    total_return_pct: float = 0.0
    n_years: float = 0.0
    n_spike_days: int = 0
    n_eligible_days: int = 0
    is_cagr: float = 0.0
    is_sharpe: float = 0.0
    oos_cagr: float = 0.0
    oos_sharpe: float = 0.0
    yearly: Dict[int, Dict] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    date_range: str = ""
    data_source: str = "IronVault options_cache.db (real)"


# ─── SPY daily close loader (real Yahoo via existing helper) ──────────────
def load_spy_daily() -> Dict[str, float]:
    from backtest.backtester import _yf_download_safe
    df = _yf_download_safe(TICKER, START_DATE, END_DATE)
    if df.empty:
        raise RuntimeError(f"Failed to load real SPY data")
    return {d.strftime("%Y-%m-%d"): float(c) for d, c in df["Close"].items()}


# ─── Find ATM expiration (20-45 DTE monthly) ───────────────────────────────
def find_atm_expiration(hd: IronVault, date_str: str) -> Optional[str]:
    """Find the next front-month expiration in DTE window."""
    import sqlite3
    conn = sqlite3.connect(hd._db_path)
    try:
        cur = conn.cursor()
        today = datetime.strptime(date_str, "%Y-%m-%d").date()
        cur.execute("""
            SELECT DISTINCT expiration FROM option_contracts
            WHERE ticker=? AND expiration > ? AND expiration <= ?
            ORDER BY expiration
        """, (TICKER,
              (today + timedelta(days=TARGET_DTE_MIN)).strftime("%Y-%m-%d"),
              (today + timedelta(days=TARGET_DTE_MAX)).strftime("%Y-%m-%d")))
        exps = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    if not exps:
        return None

    # Prefer exp closest to 30 DTE
    target = today + timedelta(days=TARGET_DTE_PREFERRED)
    exps_with_dte = [(abs((datetime.strptime(e, "%Y-%m-%d").date() - target).days), e)
                      for e in exps]
    exps_with_dte.sort()
    return exps_with_dte[0][1]


# ─── Compute IV proxy for one day ──────────────────────────────────────────
def compute_iv_proxy(hd: IronVault, date_str: str,
                     spy_price: float) -> Optional[IVObservation]:
    """Compute ATM straddle-price / SPY ratio as an IV proxy.

    REAL DATA: uses get_spread_prices / get_contract_price on real IronVault
    option chain to find the ATM put and call front-month prices.
    """
    exp = find_atm_expiration(hd, date_str)
    if not exp:
        return None

    # Find ATM strike (nearest $1 to SPY close)
    target_strike = round(spy_price)

    # Get available strikes
    put_strikes = hd.get_available_strikes(TICKER, exp, date_str, "P")
    call_strikes = hd.get_available_strikes(TICKER, exp, date_str, "C")
    if not put_strikes or not call_strikes:
        return None

    # Find closest strike available in both
    common = sorted(set(put_strikes) & set(call_strikes))
    if not common:
        return None
    atm_strike = min(common, key=lambda s: abs(s - target_strike))

    # Build OCC symbols and fetch real daily close prices
    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
    put_sym = IronVault.build_occ_symbol(TICKER, exp_dt, atm_strike, "P")
    call_sym = IronVault.build_occ_symbol(TICKER, exp_dt, atm_strike, "C")

    put_price = hd.get_contract_price(put_sym, date_str)
    call_price = hd.get_contract_price(call_sym, date_str)

    if put_price is None or call_price is None or put_price <= 0 or call_price <= 0:
        return None

    iv_proxy = (put_price + call_price) / spy_price

    return IVObservation(
        date=date_str,
        spy_price=spy_price,
        atm_strike=atm_strike,
        expiration=exp,
        put_price=put_price,
        call_price=call_price,
        iv_proxy=iv_proxy,
    )


# ─── Build full IV proxy time series ───────────────────────────────────────
def build_iv_series(hd: IronVault,
                     spy_daily: Dict[str, float]) -> List[IVObservation]:
    """For every trading day with SPY data, compute IV proxy."""
    observations = []
    dates = sorted(spy_daily.keys())
    for i, d in enumerate(dates):
        obs = compute_iv_proxy(hd, d, spy_daily[d])
        if obs is not None:
            observations.append(obs)
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(dates)} days, {len(observations)} observations")

    return observations


def detect_spikes(observations: List[IVObservation],
                   lookback: int = IV_LOOKBACK,
                   z_threshold: float = IV_SPIKE_Z) -> List[IVObservation]:
    """Compute rolling mean/std and flag spikes (CAUSAL — no look-ahead)."""
    iv_vals = [o.iv_proxy for o in observations]
    for i, obs in enumerate(observations):
        if i < lookback:
            continue
        window = iv_vals[i - lookback:i]  # causal: EXCLUDES today
        if len(window) < lookback:
            continue
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std < 1e-9:
            continue
        z = (obs.iv_proxy - mean) / std
        obs.iv_ma = round(mean, 6)
        obs.iv_std = round(std, 6)
        obs.iv_z = round(z, 3)
        obs.is_spike = (z >= z_threshold and obs.iv_proxy >= IV_MIN_LEVEL)
    return observations


# ─── Spread construction ───────────────────────────────────────────────────
def select_bull_put_spread(hd: IronVault, date_str: str, spy_price: float,
                             expiration: str) -> Optional[Dict]:
    """Pick a bull put spread: sell 3% OTM put, buy $5 lower put.

    Uses REAL IronVault option prices.
    """
    target_short = spy_price * (1 - OTM_PCT)
    put_strikes = hd.get_available_strikes(TICKER, expiration, date_str, "P")
    if not put_strikes:
        return None

    # Find short strike closest to 3% OTM
    short = min(put_strikes, key=lambda s: abs(s - target_short))
    # Long strike: $5 below short (must exist in chain)
    long_candidates = [s for s in put_strikes if abs(s - (short - SPREAD_WIDTH)) < 0.51]
    if not long_candidates:
        # Fallback: closest available strike $4-$6 below
        wider = [s for s in put_strikes if (short - 6.0) <= s < short - 3.0]
        if not wider:
            return None
        long = max(wider)
    else:
        long = long_candidates[0]

    width = short - long
    if width <= 0:
        return None

    # Get real credit from IronVault
    prices = hd.get_spread_prices(
        TICKER, datetime.strptime(expiration, "%Y-%m-%d"),
        short, long, "P", date_str,
    )
    if prices is None:
        return None

    credit = prices["short_close"] - prices["long_close"]
    if credit <= 0 or credit < width * MIN_CREDIT_PCT:
        return None

    return {
        "short_strike": short,
        "long_strike": long,
        "width": width,
        "credit": credit,
        "max_loss": width - credit,
        "expiration": expiration,
    }


def track_spread_exit(hd: IronVault, trade: Dict,
                        entry_credit: float) -> Tuple[str, str, float, int]:
    """Walk forward day by day until exit conditions trigger.

    Returns (exit_date, reason, exit_debit, hold_days).
    """
    entry_dt = datetime.strptime(trade["entry_date"], "%Y-%m-%d").date()
    exp_dt = datetime.strptime(trade["expiration"], "%Y-%m-%d").date()

    cur_dt = entry_dt + timedelta(days=1)
    hold = 0
    while cur_dt <= exp_dt and hold < MAX_HOLD_DAYS:
        date_str = cur_dt.strftime("%Y-%m-%d")
        # Skip weekends
        if cur_dt.weekday() >= 5:
            cur_dt += timedelta(days=1)
            continue
        hold += 1
        dte = (exp_dt - cur_dt).days

        prices = hd.get_spread_prices(
            TICKER, datetime.strptime(trade["expiration"], "%Y-%m-%d"),
            trade["short_strike"], trade["long_strike"], "P", date_str,
        )
        if prices is None:
            cur_dt += timedelta(days=1)
            continue

        close_debit = prices["short_close"] - prices["long_close"]

        # Profit target
        if close_debit <= entry_credit * (1 - PROFIT_TARGET_PCT):
            return date_str, "profit_target", close_debit, hold
        # Stop loss
        if close_debit - entry_credit >= entry_credit * STOP_LOSS_MULT:
            return date_str, "stop_loss", close_debit, hold
        # DTE exit
        if dte <= MIN_DTE_EXIT:
            return date_str, "dte_exit", close_debit, hold

        cur_dt += timedelta(days=1)

    # Expiration or max hold
    final_prices = hd.get_spread_prices(
        TICKER, datetime.strptime(trade["expiration"], "%Y-%m-%d"),
        trade["short_strike"], trade["long_strike"], "P", trade["expiration"],
    )
    final_debit = 0.0 if final_prices is None else (
        final_prices["short_close"] - final_prices["long_close"]
    )
    return trade["expiration"], "expiration", final_debit, hold


# ─── Main backtest ─────────────────────────────────────────────────────────
def run_backtest() -> BacktestResult:
    logger.info("=" * 70)
    logger.info("EXP-1840: IV Spike Entry")
    logger.info("=" * 70)

    hd = IronVault.instance()

    # 1. Load SPY daily
    logger.info("Loading SPY daily closes...")
    spy_daily = load_spy_daily()
    logger.info(f"  SPY: {len(spy_daily)} days")

    # 2. Build IV proxy series
    logger.info("Building IV proxy series from real ATM option chains...")
    observations = build_iv_series(hd, spy_daily)
    logger.info(f"  {len(observations)} observations with IV proxy")

    # 3. Detect spikes
    observations = detect_spikes(observations)
    n_eligible = sum(1 for o in observations if o.iv_z is not None)
    n_spikes = sum(1 for o in observations if o.is_spike)
    logger.info(f"  {n_eligible} eligible days, {n_spikes} spikes detected (z >= {IV_SPIKE_Z})")

    # 4. Execute trades on spike days
    logger.info("Executing trades on spike days...")
    trades: List[Trade] = []
    last_entry: Optional[date] = None

    for obs in observations:
        if not obs.is_spike:
            continue
        today = datetime.strptime(obs.date, "%Y-%m-%d").date()
        if last_entry and (today - last_entry).days < MIN_SPACING_DAYS:
            continue

        spread = select_bull_put_spread(hd, obs.date, obs.spy_price, obs.expiration)
        if spread is None:
            continue

        # Size: 2% risk / max_loss_per_contract
        max_loss_per_contract = spread["max_loss"] * 100
        if max_loss_per_contract <= 0:
            continue
        contracts = max(1, int(CAPITAL * RISK_PCT / max_loss_per_contract))
        contracts = min(contracts, 10)

        entry_credit = spread["credit"]
        trade_dict = {
            "entry_date": obs.date,
            "expiration": spread["expiration"],
            "short_strike": spread["short_strike"],
            "long_strike": spread["long_strike"],
        }

        exit_date, reason, exit_debit, hold_days = track_spread_exit(
            hd, trade_dict, entry_credit,
        )

        pnl = (entry_credit - exit_debit) * 100 * contracts

        trades.append(Trade(
            entry_date=obs.date,
            exit_date=exit_date,
            expiration=spread["expiration"],
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            entry_credit=round(entry_credit, 4),
            exit_debit=round(exit_debit, 4),
            contracts=contracts,
            pnl=round(pnl, 2),
            hold_days=hold_days,
            exit_reason=reason,
            iv_z_at_entry=round(obs.iv_z, 3),
            iv_proxy_at_entry=round(obs.iv_proxy, 5),
        ))
        last_entry = today

    logger.info(f"  {len(trades)} trades executed")

    # 5. Compute metrics
    if not trades:
        return BacktestResult(
            n_spike_days=n_spikes, n_eligible_days=n_eligible,
            date_range=f"{observations[0].date} -> {observations[-1].date}",
        )

    pnls = np.array([t.pnl for t in trades])
    total_pnl = float(pnls.sum())
    wins = int((pnls > 0).sum())

    start_dt = datetime.strptime(trades[0].entry_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(trades[-1].exit_date, "%Y-%m-%d").date()
    n_years = max((end_dt - start_dt).days / 365.25, 0.5)
    total_return = total_pnl / CAPITAL
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Arithmetic per-trade Sharpe
    if len(pnls) > 1:
        mean_pnl = float(pnls.mean())
        std_pnl = float(pnls.std(ddof=1))
        trades_per_year = len(pnls) / n_years
        sharpe = (mean_pnl / std_pnl * math.sqrt(trades_per_year)) if std_pnl > 1e-9 else 0.0
    else:
        sharpe = 0.0

    # Max DD from trade cumulative curve
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum)
    max_dd = float(dd.max() / CAPITAL) if len(dd) > 0 else 0.0

    # Walk-forward IS/OOS
    is_pnls = [t.pnl for t in trades
               if int(t.exit_date[:4]) <= IS_END_YEAR]
    oos_pnls = [t.pnl for t in trades
                if int(t.exit_date[:4]) >= OOS_START_YEAR]

    def _subset_metrics(sub_pnls):
        if len(sub_pnls) < 2:
            return 0.0, 0.0
        arr = np.array(sub_pnls)
        yrs_span = max(len(set(int(t.exit_date[:4])
                               for t in trades
                               if t.pnl in sub_pnls)), 1)
        tr_per_yr = len(arr) / yrs_span
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        s = mean / std * math.sqrt(tr_per_yr) if std > 1e-9 else 0.0
        c = (1 + arr.sum() / CAPITAL) ** (1 / yrs_span) - 1
        return c, s

    is_cagr, is_sharpe = _subset_metrics(is_pnls)
    oos_cagr, oos_sharpe = _subset_metrics(oos_pnls)

    # Yearly breakdown
    yearly = {}
    for t in trades:
        yr = int(t.exit_date[:4])
        yearly.setdefault(yr, []).append(t.pnl)
    yearly_stats = {}
    for yr, ps in sorted(yearly.items()):
        arr = np.array(ps)
        w = float((arr > 0).sum() / len(arr))
        yearly_stats[yr] = {
            "n": len(arr),
            "pnl": round(float(arr.sum()), 2),
            "wr": round(w, 4),
            "return_pct": round(float(arr.sum() / CAPITAL * 100), 2),
        }

    return BacktestResult(
        n_trades=len(trades),
        total_pnl=round(total_pnl, 2),
        win_rate=round(wins / len(trades), 4),
        avg_pnl_per_trade=round(total_pnl / len(trades), 2),
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 3),
        max_dd=round(max_dd, 4),
        total_return_pct=round(total_return * 100, 2),
        n_years=round(n_years, 2),
        n_spike_days=n_spikes,
        n_eligible_days=n_eligible,
        is_cagr=round(is_cagr, 4),
        is_sharpe=round(is_sharpe, 3),
        oos_cagr=round(oos_cagr, 4),
        oos_sharpe=round(oos_sharpe, 3),
        yearly=yearly_stats,
        trades=trades,
        date_range=f"{observations[0].date} -> {observations[-1].date}",
    )


def compare_to_exp1220(result: BacktestResult) -> Dict:
    """Load EXP-1220 base metrics for comparison."""
    try:
        d = json.load(open(ROOT / "reports" / "exp1220_dynamic_leverage.json"))
        base = d.get("static_1_2x", {})
        return {
            "exp1220_base": {
                "cagr_pct": base.get("cagr_pct"),
                "sharpe": base.get("sharpe"),
                "max_dd_pct": base.get("max_dd_pct"),
                "vol_pct": base.get("vol_pct"),
                "source": "reports/exp1220_dynamic_leverage.json (static_1_2x)",
            },
            "exp1840": {
                "cagr_pct": result.cagr * 100,
                "sharpe": result.sharpe,
                "max_dd_pct": result.max_dd * 100,
                "n_trades": result.n_trades,
                "win_rate": result.win_rate,
            },
            "sharpe_improvement": round(
                result.sharpe - base.get("sharpe", 0), 3
            ),
            "verdict": (
                "EXP-1840 IV-spike timing IMPROVES Sharpe vs EXP-1220 base"
                if result.sharpe > base.get("sharpe", 0)
                else "EXP-1840 does NOT improve Sharpe vs EXP-1220 base"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def save_results(result: BacktestResult, comparison: Dict):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # JSON
    json_path = reports_dir / "exp1840_iv_spike_entry.json"
    payload = {
        "experiment": "EXP-1840",
        "name": "IV Spike Entry — Volatility-Timed Credit Spreads",
        "data_source": result.data_source,
        "date_range": result.date_range,
        "rule_zero_compliant": True,
        "generated": datetime.now().isoformat(),
        "parameters": {
            "ticker": TICKER,
            "iv_lookback": IV_LOOKBACK,
            "iv_spike_z": IV_SPIKE_Z,
            "target_dte_min": TARGET_DTE_MIN,
            "target_dte_max": TARGET_DTE_MAX,
            "spread_width": SPREAD_WIDTH,
            "otm_pct": OTM_PCT,
            "min_credit_pct": MIN_CREDIT_PCT,
            "profit_target_pct": PROFIT_TARGET_PCT,
            "stop_loss_mult": STOP_LOSS_MULT,
            "risk_pct": RISK_PCT,
            "min_spacing_days": MIN_SPACING_DAYS,
        },
        "signal_stats": {
            "eligible_days": result.n_eligible_days,
            "spike_days": result.n_spike_days,
            "spike_rate": (round(result.n_spike_days / max(result.n_eligible_days, 1), 4)),
        },
        "metrics": {
            "n_trades": result.n_trades,
            "total_pnl": result.total_pnl,
            "win_rate": result.win_rate,
            "avg_pnl_per_trade": result.avg_pnl_per_trade,
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "max_dd": result.max_dd,
            "total_return_pct": result.total_return_pct,
            "n_years": result.n_years,
        },
        "walk_forward": {
            "is_period": f"2020-{IS_END_YEAR}",
            "oos_period": f"{OOS_START_YEAR}-2025",
            "is_cagr": result.is_cagr,
            "is_sharpe": result.is_sharpe,
            "oos_cagr": result.oos_cagr,
            "oos_sharpe": result.oos_sharpe,
        },
        "yearly": result.yearly,
        "comparison": comparison,
        "trades": [t.__dict__ for t in result.trades],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"JSON report: {json_path}")

    # HTML
    html_path = reports_dir / "exp1840_iv_spike_entry.html"
    html_path.write_text(_build_html(result, comparison), encoding="utf-8")
    logger.info(f"HTML report: {html_path}")


def _build_html(r: BacktestResult, comp: Dict) -> str:
    yearly_rows = ""
    for yr in sorted(r.yearly.keys()):
        y = r.yearly[yr]
        color = "#059669" if y["pnl"] > 0 else "#dc2626"
        yearly_rows += (
            f'<tr><td>{yr}</td>'
            f'<td class="r">{y["n"]}</td>'
            f'<td class="r" style="color:{color}">${y["pnl"]:,.0f}</td>'
            f'<td class="r">{y["wr"]:.0%}</td>'
            f'<td class="r" style="color:{color}">{y["return_pct"]:+.2f}%</td></tr>\n'
        )

    trade_rows = ""
    for t in r.trades[:30]:
        c = "#059669" if t.pnl > 0 else "#dc2626"
        trade_rows += (
            f'<tr><td>{t.entry_date}</td><td>{t.exit_date}</td>'
            f'<td class="r">{t.iv_z_at_entry:+.2f}</td>'
            f'<td class="r">{t.short_strike:.0f}/{t.long_strike:.0f}</td>'
            f'<td class="r">${t.entry_credit:.2f}</td>'
            f'<td class="r">{t.contracts}</td>'
            f'<td class="r">{t.hold_days}</td>'
            f'<td class="r" style="color:{c}">${t.pnl:,.0f}</td>'
            f'<td>{t.exit_reason}</td></tr>\n'
        )

    cagr_c = "#059669" if r.cagr > 0 else "#dc2626"
    sharpe_c = "#059669" if r.sharpe > 1 else ("#d97706" if r.sharpe > 0 else "#dc2626")

    base = comp.get("exp1220_base", {}) or {}
    ex1840 = comp.get("exp1840", {}) or {}
    improvement = comp.get("sharpe_improvement", 0) or 0
    verdict = comp.get("verdict", "comparison unavailable")
    verdict_color = "#059669" if "IMPROVES" in verdict else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EXP-1840: IV Spike Entry</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e2e8f0;--text:#1a1a2e;--muted:#64748b;--green:#059669;--red:#dc2626;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;max-width:1100px;margin:0 auto;padding:28px}}
h1{{font-size:1.55rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:32px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
.sub{{color:var(--muted);font-size:.86rem;margin-bottom:18px}}
.note{{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem}}
th{{background:#f1f5f9;color:var(--muted);padding:7px 10px;text-align:left;border-bottom:2px solid var(--border);font-size:.74rem;font-weight:600;text-transform:uppercase}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}
.hero{{background:linear-gradient(135deg,#f1f5f9,#e2e8f0);border:2px solid {verdict_color};border-radius:12px;padding:24px;margin:18px 0;text-align:center}}
.hero .big{{font-size:1.5rem;font-weight:800;color:{verdict_color};margin:6px 0}}
.hero p{{color:#475569;font-size:.9rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:13px;text-align:center}}
.c .l{{color:var(--muted);font-size:.72rem;text-transform:uppercase}}
.c .v{{font-weight:700;font-size:1.1rem;margin-top:3px}}
.box{{border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0;background:var(--card)}}
.box-green{{border-left:5px solid var(--green)}} .box-red{{border-left:5px solid var(--red)}}
.box-blue{{border-left:5px solid var(--blue)}}
.box h4{{margin:0 0 6px;font-size:.95rem}}
</style></head><body>

<h1>EXP-1840: IV Spike Entry</h1>
<p class="sub">Volatility-timed credit spread overlay on SPY &bull;
Real IronVault data {r.date_range} &bull; {datetime.now().strftime("%Y-%m-%d")}</p>

<div class="hero">
<div class="big">{verdict}</div>
<p>EXP-1840 Sharpe: {ex1840.get('sharpe', 0):.2f} &bull;
EXP-1220 base Sharpe: {base.get('sharpe', 0):.2f} &bull;
&Delta; {improvement:+.3f}</p>
</div>

<div class="cards">
<div class="c"><div class="l">Trades</div><div class="v">{r.n_trades}</div></div>
<div class="c"><div class="l">Spike Days</div><div class="v">{r.n_spike_days} / {r.n_eligible_days}</div></div>
<div class="c"><div class="l">Win Rate</div><div class="v">{r.win_rate:.0%}</div></div>
<div class="c"><div class="l">Total PnL</div><div class="v" style="color:{cagr_c}">${r.total_pnl:,.0f}</div></div>
<div class="c"><div class="l">CAGR</div><div class="v" style="color:{cagr_c}">{r.cagr*100:+.2f}%</div></div>
<div class="c"><div class="l">Sharpe</div><div class="v" style="color:{sharpe_c}">{r.sharpe:.2f}</div></div>
<div class="c"><div class="l">Max DD</div><div class="v">{r.max_dd*100:.2f}%</div></div>
<div class="c"><div class="l">Years</div><div class="v">{r.n_years:.1f}</div></div>
</div>

<h2>1. Strategy Logic</h2>
<div class="box box-blue">
<h4>Hypothesis</h4>
<p>Credit spread sellers are paid to absorb volatility risk. Instead of entering on a
fixed cadence (EXP-1220's 7d), time entries to moments when IV spikes above its 5-day
rolling mean by &ge; 1.5&sigma;. These moments offer the richest premium.</p>
<h4 style="margin-top:12px">Signal</h4>
<ul style="padding-left:20px;font-size:.85rem;line-height:1.85;margin-top:4px">
<li>IV proxy = (ATM put + ATM call) / SPY, from real IronVault front-month prices</li>
<li>Rolling 5-day mean and std (causal — excludes today)</li>
<li>Spike: z = (today - mean) / std &ge; {IV_SPIKE_Z}</li>
<li>Enter bull put spread: sell {OTM_PCT*100:.0f}% OTM, buy ${SPREAD_WIDTH:.0f} lower</li>
<li>Exit: 50% profit, 2&times; stop, or 5 DTE</li>
</ul>
</div>

<h2>2. Comparison vs EXP-1220 Base Strategy</h2>
<table>
<thead><tr><th>Metric</th><th class="r">EXP-1220 (1.2x base)</th><th class="r">EXP-1840 (IV spike)</th><th class="r">&Delta;</th></tr></thead>
<tbody>
<tr><td>CAGR</td><td class="r">{base.get('cagr_pct', 0):.2f}%</td><td class="r">{ex1840.get('cagr_pct', 0):.2f}%</td><td class="r">{(ex1840.get('cagr_pct',0) or 0) - (base.get('cagr_pct',0) or 0):+.2f}%</td></tr>
<tr><td>Sharpe</td><td class="r">{base.get('sharpe', 0):.2f}</td><td class="r">{ex1840.get('sharpe', 0):.2f}</td><td class="r">{improvement:+.3f}</td></tr>
<tr><td>Max DD</td><td class="r">{base.get('max_dd_pct', 0):.2f}%</td><td class="r">{ex1840.get('max_dd_pct', 0):.2f}%</td><td class="r">&mdash;</td></tr>
</tbody></table>

<h2>3. Walk-Forward Validation</h2>
<table>
<thead><tr><th>Period</th><th class="r">CAGR</th><th class="r">Sharpe</th></tr></thead>
<tbody>
<tr><td>In-Sample (2020-{IS_END_YEAR})</td><td class="r">{r.is_cagr*100:+.2f}%</td><td class="r">{r.is_sharpe:.2f}</td></tr>
<tr><td>Out-of-Sample ({OOS_START_YEAR}-2025)</td><td class="r">{r.oos_cagr*100:+.2f}%</td><td class="r">{r.oos_sharpe:.2f}</td></tr>
</tbody></table>

<h2>4. Yearly Performance</h2>
<table>
<thead><tr><th>Year</th><th class="r">Trades</th><th class="r">PnL</th><th class="r">Win Rate</th><th class="r">Return %</th></tr></thead>
<tbody>{yearly_rows}</tbody>
</table>

<h2>5. Recent Trades (first 30)</h2>
<table>
<thead><tr><th>Entry</th><th>Exit</th><th class="r">IV z</th><th class="r">Strikes</th><th class="r">Credit</th><th class="r">Qty</th><th class="r">Hold</th><th class="r">PnL</th><th>Reason</th></tr></thead>
<tbody>{trade_rows}</tbody>
</table>

<h2>6. Rule Zero Compliance</h2>
<div class="box box-green">
<h4>ZERO SYNTHETIC DATA</h4>
<ul style="padding-left:20px;font-size:.82rem">
<li>IV proxy: real IronVault ATM front-month option prices</li>
<li>Spread entry/exit: real IronVault get_spread_prices for every leg</li>
<li>SPY daily: real Yahoo Finance closes</li>
<li>No np.random. No Black-Scholes. No fabricated IV values.</li>
</ul>
</div>

<p style="text-align:center;color:var(--muted);margin-top:36px;padding-top:14px;border-top:1px solid var(--border);font-size:.78rem">
EXP-1840 IV Spike Entry &bull; compass/iv_spike_entry.py &bull;
{datetime.now().strftime("%Y-%m-%d")}
</p>
</body></html>"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                         datefmt="%H:%M:%S")
    result = run_backtest()
    comparison = compare_to_exp1220(result)
    save_results(result, comparison)

    print(f"\n{'=' * 70}")
    print(f"EXP-1840: IV Spike Entry — Summary")
    print(f"{'=' * 70}")
    print(f"Trades:         {result.n_trades}")
    print(f"Spike days:     {result.n_spike_days} / {result.n_eligible_days} eligible")
    print(f"Win rate:       {result.win_rate:.1%}")
    print(f"Total PnL:      ${result.total_pnl:,.2f}")
    print(f"CAGR:           {result.cagr*100:+.2f}%")
    print(f"Sharpe:         {result.sharpe:.2f}")
    print(f"Max DD:         {result.max_dd*100:.2f}%")
    print(f"Years:          {result.n_years:.1f}")
    print(f"\nWalk-forward:")
    print(f"  IS (2020-{IS_END_YEAR}):  CAGR {result.is_cagr*100:+.2f}%, Sharpe {result.is_sharpe:.2f}")
    print(f"  OOS ({OOS_START_YEAR}-2025): CAGR {result.oos_cagr*100:+.2f}%, Sharpe {result.oos_sharpe:.2f}")
    print(f"\nComparison vs EXP-1220 base:")
    base = comparison.get("exp1220_base", {})
    ex1840 = comparison.get("exp1840", {})
    if base:
        print(f"  EXP-1220:  CAGR {base.get('cagr_pct', 0):.2f}%, Sharpe {base.get('sharpe', 0):.2f}")
        print(f"  EXP-1840:  CAGR {ex1840.get('cagr_pct', 0):.2f}%, Sharpe {ex1840.get('sharpe', 0):.2f}")
        print(f"  Sharpe delta: {comparison.get('sharpe_improvement', 0):+.3f}")
        print(f"\n  VERDICT: {comparison.get('verdict')}")
    print(f"{'=' * 70}")
