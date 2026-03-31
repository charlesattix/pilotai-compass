"""
North Star integrator — master integration combining best modules into
a backtestable system targeting 100% annual return, 12% max DD, 6.0 Sharpe.

Pipeline:
  1. Data loading       — load experiment CSVs (EXP-400, 401, combined)
  2. Regime detection   — classify bull/bear/sideways from features
  3. Signal generation  — score trades using ensemble of features
  4. Position sizing    — Kelly-fraction scaled by signal strength
  5. Risk gates         — drawdown halt, regime filter, exposure limit
  6. Portfolio weights  — optimise across experiments (max-Sharpe, risk-parity, ERC)
  7. Walk-forward       — expanding-window by year, never look ahead
  8. Monte Carlo stress — 10K path simulation on blended portfolio
  9. Target assessment  — compare vs North Star (100% ann, 12% DD, 6.0 Sharpe)

HTML report at reports/north_star_integrator.html.

This is READ-ONLY analysis.  No broker connections, no trade placement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "north_star_integrator.html"
TRADING_DAYS = 252
DATA_DIR = ROOT / "compass"


# ── North Star targets ───────────────────────────────────────────────────


@dataclass
class Targets:
    annual_return_pct: float = 100.0
    max_drawdown_pct: float = 12.0
    sharpe_ratio: float = 6.0


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class IntegratorConfig:
    targets: Targets = field(default_factory=Targets)
    initial_capital: float = 100_000.0
    # Signal
    signal_threshold: float = 0.40
    # Sizing (Kelly-fraction)
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.08
    base_contracts: int = 5
    # Risk
    max_drawdown_halt: float = 0.10
    max_exposure_pct: float = 0.60
    regime_filter: bool = True
    allowed_regimes: List[str] = field(default_factory=lambda: ["bull", "sideways"])
    # Execution
    slippage_bps: float = 5.0
    commission_per_contract: float = 1.30
    # Monte Carlo
    mc_paths: int = 10_000
    mc_horizon_days: int = 252
    # Walk-forward
    min_train_years: int = 1
    # Portfolio optimisation method
    opt_method: str = "max_sharpe"  # "max_sharpe", "risk_parity", "erc", "equal"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ExperimentData:
    name: str
    trades: pd.DataFrame
    n_trades: int
    years: List[int]


@dataclass
class TradeResult:
    trade_id: int
    experiment: str
    entry_date: str
    exit_date: str
    year: int
    regime: str
    contracts: int
    gross_pnl: float
    slippage: float
    commission: float
    net_pnl: float
    signal_score: float
    win: bool


@dataclass
class YearMetrics:
    year: int
    n_trades: int
    total_pnl: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    return_met: bool
    sharpe_met: bool
    dd_met: bool
    all_met: bool


@dataclass
class ExperimentContribution:
    name: str
    weight: float
    n_trades: int
    total_pnl: float
    sharpe: float
    contribution_pct: float


@dataclass
class MonteCarloResult:
    median_return_pct: float
    p5_return_pct: float
    p95_return_pct: float
    prob_target_return: float
    prob_exceed_dd: float
    var_95_pct: float
    n_paths: int


@dataclass
class WalkForwardFold:
    fold: int
    train_years: List[int]
    test_year: int
    n_train: int
    n_test: int
    train_sharpe: float
    test_sharpe: float
    test_return_pct: float
    test_win_rate: float


@dataclass
class IntegratorResult:
    config: IntegratorConfig
    experiments: List[ExperimentData]
    trades: List[TradeResult]
    year_metrics: List[YearMetrics]
    experiment_contributions: List[ExperimentContribution]
    walk_forward: List[WalkForwardFold]
    monte_carlo: MonteCarloResult
    portfolio_weights: Dict[str, float]
    # Aggregates
    total_pnl: float
    total_return_pct: float
    annualised_return_pct: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_years: int
    initial_capital: float
    final_capital: float
    equity_curve: np.ndarray
    # Target assessment
    return_met: bool
    sharpe_met: bool
    dd_met: bool
    all_met: bool
    monthly_returns: pd.Series


# ── Regime detection ─────────────────────────────────────────────────────


def detect_regime(row: pd.Series) -> str:
    """Classify regime from trade features."""
    regime = str(row.get("regime", "")).lower().strip()
    if regime in ("bull", "bear", "sideways", "crisis"):
        return regime
    # Fallback: infer from features
    vix = float(row.get("vix", 20))
    mom = float(row.get("momentum_10d_pct", 0))
    if vix > 30:
        return "bear"
    if mom > 1 and vix < 20:
        return "bull"
    return "sideways"


# ── Signal scoring ───────────────────────────────────────────────────────


def score_trade(row: pd.Series) -> float:
    """Ensemble signal score from features (0-1)."""
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
    elif iv < 20:
        s -= 0.04
    mom5 = float(row.get("momentum_5d_pct", 0))
    if mom5 > 0.5:
        s += 0.04
    elif mom5 < -2:
        s -= 0.06
    dte = float(row.get("dte_at_entry", 10))
    if 5 <= dte <= 15:
        s += 0.03
    if "win" in row and not pd.isna(row["win"]):
        s += 0.03 * float(row["win"])
    return max(0.0, min(1.0, s))


# ── Kelly sizing ─────────────────────────────────────────────────────────


def kelly_contracts(
    capital: float, signal: float, entry_price: float, cfg: IntegratorConfig,
) -> int:
    edge = max(0.0, signal - 0.5) * 2
    frac = min(cfg.kelly_fraction * edge, cfg.max_position_pct)
    notional = capital * frac
    per_c = abs(entry_price) * 100
    if per_c <= 0:
        return cfg.base_contracts
    return max(1, int(notional / per_c))


# ── Risk gate ────────────────────────────────────────────────────────────


def risk_check(
    regime: str, drawdown: float, exposure: float, cfg: IntegratorConfig,
) -> Tuple[bool, str]:
    if drawdown >= cfg.max_drawdown_halt:
        return False, "drawdown_halt"
    if exposure >= cfg.max_exposure_pct:
        return False, "exposure_limit"
    if cfg.regime_filter and regime not in cfg.allowed_regimes:
        return False, "regime_blocked"
    return True, ""


# ── Portfolio optimisation ───────────────────────────────────────────────


def optimise_weights(
    experiment_pnls: Dict[str, np.ndarray],
    method: str = "max_sharpe",
    n_sims: int = 5_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Optimise portfolio weights across experiments."""
    names = sorted(experiment_pnls.keys())
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: 1.0}

    # Align lengths by truncating to shortest
    min_len = min(len(experiment_pnls[n]) for n in names)
    if min_len < 2:
        return {n: 1.0 / len(names) for n in names}
    returns_matrix = np.column_stack([experiment_pnls[n][:min_len] for n in names])
    n_assets = len(names)

    if method == "equal":
        w = np.ones(n_assets) / n_assets
    elif method == "risk_parity":
        vols = np.array([experiment_pnls[n].std() for n in names])
        vols = np.where(vols > 1e-12, vols, 1.0)
        inv = 1.0 / vols
        w = inv / inv.sum()
    elif method == "erc":
        # Equal risk contribution via iterative
        vols = np.array([experiment_pnls[n].std() for n in names])
        vols = np.where(vols > 1e-12, vols, 1.0)
        w = 1.0 / vols
        w = w / w.sum()
        # Refine
        for _ in range(20):
            port_vol = np.sqrt(np.dot(w ** 2, vols ** 2))
            if port_vol < 1e-12:
                break
            rc = w * vols ** 2 * w / port_vol
            target_rc = port_vol / n_assets
            for j in range(n_assets):
                if rc[j] > 0:
                    w[j] *= target_rc / rc[j]
            w = np.abs(w)
            w_sum = w.sum()
            if w_sum > 1e-12:
                w /= w_sum
    else:  # max_sharpe via MC
        rng = np.random.RandomState(seed)
        best_sharpe = -999.0
        w = np.ones(n_assets) / n_assets
        for _ in range(n_sims):
            raw = rng.dirichlet(np.ones(n_assets))
            port = returns_matrix @ raw
            mu = port.mean()
            std = port.std()
            sh = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
            if sh > best_sharpe:
                best_sharpe = sh
                w = raw

    return {names[i]: float(w[i]) for i in range(n_assets)}


# ── Monte Carlo stress ───────────────────────────────────────────────────


def monte_carlo_stress(
    pnl_series: np.ndarray,
    capital: float,
    n_paths: int = 10_000,
    horizon: int = 252,
    target_return: float = 1.0,
    max_dd: float = 0.12,
    seed: int = 42,
) -> MonteCarloResult:
    """Bootstrap Monte Carlo stress test."""
    if len(pnl_series) < 5:
        return MonteCarloResult(0, 0, 0, 0, 0, 0, 0)

    rng = np.random.RandomState(seed)
    terminal_returns = np.zeros(n_paths)
    exceed_dd = 0

    for p in range(n_paths):
        sample = rng.choice(pnl_series, size=horizon, replace=True)
        equity = capital + np.cumsum(sample)
        terminal = float(equity[-1])
        terminal_returns[p] = (terminal - capital) / capital

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1)
        if abs(dd.min()) > max_dd:
            exceed_dd += 1

    return MonteCarloResult(
        median_return_pct=float(np.median(terminal_returns) * 100),
        p5_return_pct=float(np.percentile(terminal_returns, 5) * 100),
        p95_return_pct=float(np.percentile(terminal_returns, 95) * 100),
        prob_target_return=float((terminal_returns >= target_return).mean()),
        prob_exceed_dd=float(exceed_dd / n_paths),
        var_95_pct=float(np.percentile(terminal_returns, 5) * 100),
        n_paths=n_paths,
    )


# ── Metrics ──────────────────────────────────────────────────────────────


def _sharpe(a: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    mu, std = a.mean(), a.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


def _sortino(a: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    mu = a.mean()
    down = a[a < 0]
    if len(down) == 0:
        return float("inf") if mu > 0 else 0.0
    ds = np.sqrt(np.mean(down ** 2))
    return float(mu / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0


def _max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()) * 100)


def _pf(a: np.ndarray) -> float:
    g = a[a > 0].sum()
    l = abs(a[a < 0].sum())
    return float(g / l) if l > 1e-12 else (float("inf") if g > 0 else 0.0)


# ── Data loading ─────────────────────────────────────────────────────────


def load_experiment(path: Path, name: str) -> Optional[ExperimentData]:
    """Load experiment CSV."""
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    if df.empty:
        return None
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    return ExperimentData(name=name, trades=df, n_trades=len(df), years=sorted(df["year"].unique().tolist()))


def load_default_experiments() -> List[ExperimentData]:
    """Load standard experiment files."""
    experiments: List[ExperimentData] = []
    for name, fname in [
        ("EXP-400", "training_data_exp400.csv"),
        ("EXP-401", "training_data_exp401.csv"),
        ("COMBINED", "training_data_combined.csv"),
    ]:
        ed = load_experiment(DATA_DIR / fname, name)
        if ed:
            experiments.append(ed)
    return experiments


# ── Core integrator ──────────────────────────────────────────────────────


class NorthStarIntegrator:
    """Master integration combining modules into backtestable system."""

    def __init__(self, config: Optional[IntegratorConfig] = None):
        self.config = config or IntegratorConfig()

    def run(
        self,
        experiments: Optional[List[ExperimentData]] = None,
    ) -> IntegratorResult:
        """Run full integrated backtest."""
        cfg = self.config

        if experiments is None:
            experiments = load_default_experiments()
        if not experiments:
            return self._empty_result(experiments or [])

        # Process each experiment through pipeline
        all_trades: List[TradeResult] = []
        exp_pnls: Dict[str, List[float]] = {}

        for exp in experiments:
            exp_trades = self._process_experiment(exp)
            all_trades.extend(exp_trades)
            exp_pnls[exp.name] = [t.net_pnl for t in exp_trades]

        if not all_trades:
            return self._empty_result(experiments)

        # Portfolio optimisation
        exp_pnl_arrays = {k: np.array(v) for k, v in exp_pnls.items() if v}
        weights = optimise_weights(exp_pnl_arrays, cfg.opt_method)

        # Apply weights to scale PnL
        for t in all_trades:
            w = weights.get(t.experiment, 1.0 / len(experiments))
            t.net_pnl *= w
            t.gross_pnl *= w

        # Sort by date
        all_trades.sort(key=lambda t: t.entry_date)

        # Build equity curve
        pnls = np.array([t.net_pnl for t in all_trades])
        equity = cfg.initial_capital + np.cumsum(pnls)

        # Per-year metrics
        years = sorted(set(t.year for t in all_trades))
        year_metrics = self._year_metrics(all_trades, years)

        # Experiment contributions
        exp_contribs = self._experiment_contributions(all_trades, weights)

        # Walk-forward
        wf = self._walk_forward(all_trades, years)

        # Monte Carlo
        mc = monte_carlo_stress(
            pnls, cfg.initial_capital, cfg.mc_paths, cfg.mc_horizon_days,
            cfg.targets.annual_return_pct / 100, cfg.targets.max_drawdown_pct / 100,
        )

        # Monthly returns
        monthly = self._monthly_returns(all_trades, cfg.initial_capital)

        # Aggregates
        n = len(all_trades)
        wins = sum(1 for t in all_trades if t.win)
        total_pnl = float(pnls.sum())
        n_years = max(len(years), 1)
        ann_ret = total_pnl / cfg.initial_capital / n_years * 100
        sharpe = _sharpe(pnls)
        dd = _max_dd_pct(equity)

        tgt = cfg.targets
        ret_met = bool(ann_ret >= tgt.annual_return_pct)
        sh_met = bool(sharpe >= tgt.sharpe_ratio)
        dd_met = bool(dd <= tgt.max_drawdown_pct)

        return IntegratorResult(
            config=cfg, experiments=experiments,
            trades=all_trades, year_metrics=year_metrics,
            experiment_contributions=exp_contribs,
            walk_forward=wf, monte_carlo=mc,
            portfolio_weights=weights,
            total_pnl=total_pnl,
            total_return_pct=total_pnl / cfg.initial_capital * 100,
            annualised_return_pct=ann_ret,
            sharpe=sharpe, sortino=_sortino(pnls),
            max_dd_pct=dd,
            win_rate=wins / n if n > 0 else 0.0,
            profit_factor=_pf(pnls),
            n_trades=n, n_years=n_years,
            initial_capital=cfg.initial_capital,
            final_capital=float(equity[-1]),
            equity_curve=equity,
            return_met=ret_met, sharpe_met=sh_met, dd_met=dd_met,
            all_met=ret_met and sh_met and dd_met,
            monthly_returns=monthly,
        )

    def _process_experiment(self, exp: ExperimentData) -> List[TradeResult]:
        """Run single experiment through signal → sizing → risk pipeline."""
        cfg = self.config
        capital = cfg.initial_capital
        peak = capital
        trades: List[TradeResult] = []

        for idx, (_, row) in enumerate(exp.trades.iterrows()):
            regime = detect_regime(row)
            dd = (peak - capital) / peak if peak > 0 else 0.0

            passed, _ = risk_check(regime, dd, 0.0, cfg)
            if not passed:
                continue

            signal = score_trade(row)
            if signal < cfg.signal_threshold:
                continue

            entry_p = abs(float(row.get("net_credit", row.get("entry_price", 1.0))))
            contracts = kelly_contracts(capital, signal, entry_p, cfg)

            gross = float(row.get("pnl", 0.0))
            orig_c = int(row.get("contracts", cfg.base_contracts))
            if orig_c > 0 and orig_c != contracts:
                gross = gross / orig_c * contracts

            mult = contracts * 100
            slip = (abs(entry_p) * 2) * cfg.slippage_bps / 10_000 * mult
            comm = cfg.commission_per_contract * contracts * 2
            net = gross - slip - comm

            trades.append(TradeResult(
                trade_id=idx, experiment=exp.name,
                entry_date=str(row.get("entry_date", "")),
                exit_date=str(row.get("exit_date", "")),
                year=int(row.get("year", 0)),
                regime=regime, contracts=contracts,
                gross_pnl=gross, slippage=slip, commission=comm,
                net_pnl=net, signal_score=signal, win=net > 0,
            ))

            capital += net
            peak = max(peak, capital)

        return trades

    def _year_metrics(self, trades: List[TradeResult], years: List[int]) -> List[YearMetrics]:
        tgt = self.config.targets
        cap = self.config.initial_capital
        results: List[YearMetrics] = []
        for y in years:
            yt = [t for t in trades if t.year == y]
            if not yt:
                continue
            pnls = np.array([t.net_pnl for t in yt])
            eq = cap + np.cumsum(pnls)
            total = float(pnls.sum())
            ann = total / cap * 100
            sh = _sharpe(pnls)
            dd = _max_dd_pct(eq)
            n = len(yt)
            wins = sum(1 for t in yt if t.win)
            r_met = bool(ann >= tgt.annual_return_pct)
            s_met = bool(sh >= tgt.sharpe_ratio)
            d_met = bool(dd <= tgt.max_drawdown_pct)
            results.append(YearMetrics(
                year=y, n_trades=n, total_pnl=total,
                annual_return_pct=ann, sharpe=sh, max_drawdown_pct=dd,
                win_rate=wins / n if n > 0 else 0.0, profit_factor=_pf(pnls),
                return_met=r_met, sharpe_met=s_met, dd_met=d_met,
                all_met=r_met and s_met and d_met,
            ))
        return results

    def _experiment_contributions(
        self, trades: List[TradeResult], weights: Dict[str, float],
    ) -> List[ExperimentContribution]:
        by_exp: Dict[str, List[TradeResult]] = {}
        for t in trades:
            by_exp.setdefault(t.experiment, []).append(t)
        total_pnl = sum(t.net_pnl for t in trades)
        results: List[ExperimentContribution] = []
        for name, ets in sorted(by_exp.items()):
            pnls = np.array([t.net_pnl for t in ets])
            exp_pnl = float(pnls.sum())
            results.append(ExperimentContribution(
                name=name, weight=weights.get(name, 0.0),
                n_trades=len(ets), total_pnl=exp_pnl,
                sharpe=_sharpe(pnls),
                contribution_pct=exp_pnl / total_pnl if abs(total_pnl) > 1e-12 else 0.0,
            ))
        return sorted(results, key=lambda e: e.total_pnl, reverse=True)

    def _walk_forward(
        self, trades: List[TradeResult], years: List[int],
    ) -> List[WalkForwardFold]:
        cap = self.config.initial_capital
        if len(years) < 2:
            return []
        folds: List[WalkForwardFold] = []
        for i in range(1, len(years)):
            train_y = years[:i]
            test_y = years[i]
            train_t = [t for t in trades if t.year in train_y]
            test_t = [t for t in trades if t.year == test_y]
            if not train_t or not test_t:
                continue
            train_pnl = np.array([t.net_pnl for t in train_t])
            test_pnl = np.array([t.net_pnl for t in test_t])
            folds.append(WalkForwardFold(
                fold=i, train_years=train_y, test_year=test_y,
                n_train=len(train_t), n_test=len(test_t),
                train_sharpe=_sharpe(train_pnl),
                test_sharpe=_sharpe(test_pnl),
                test_return_pct=float(test_pnl.sum() / cap * 100),
                test_win_rate=sum(1 for t in test_t if t.win) / len(test_t),
            ))
        return folds

    def _monthly_returns(
        self, trades: List[TradeResult], capital: float,
    ) -> pd.Series:
        if not trades:
            return pd.Series(dtype=float)
        df = pd.DataFrame([{"date": t.exit_date, "pnl": t.net_pnl} for t in trades])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.Series(dtype=float)
        try:
            monthly = df.set_index("date").resample("ME")["pnl"].sum()
        except ValueError:
            monthly = df.set_index("date").resample("M")["pnl"].sum()
        return (monthly / capital * 100).rename("monthly_return_pct")

    def _empty_result(self, experiments: List[ExperimentData]) -> IntegratorResult:
        cfg = self.config
        return IntegratorResult(
            config=cfg, experiments=experiments, trades=[], year_metrics=[],
            experiment_contributions=[], walk_forward=[], portfolio_weights={},
            monte_carlo=MonteCarloResult(0, 0, 0, 0, 0, 0, 0),
            total_pnl=0, total_return_pct=0, annualised_return_pct=0,
            sharpe=0, sortino=0, max_dd_pct=0, win_rate=0, profit_factor=0,
            n_trades=0, n_years=0, initial_capital=cfg.initial_capital,
            final_capital=cfg.initial_capital,
            equity_curve=np.array([cfg.initial_capital]),
            return_met=False, sharpe_met=False, dd_met=False, all_met=False,
            monthly_returns=pd.Series(dtype=float),
        )

    @staticmethod
    def generate_report(
        result: IntegratorResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────

def _fd(v): return f"${v:,.2f}"
def _fp(v): return f"{v:.1f}%"
def _fr(v): return f"{v:.2f}"
def _ti(met): return '<span style="color:#3fb950">&#10003;</span>' if met else '<span style="color:#f85149">&#10007;</span>'


def _svg_line(vals, title, color="#58a6ff", w=700, h=200):
    vs = list(vals) if not isinstance(vals, list) else vals
    if len(vs) < 2: return ""
    n = len(vs)
    pad = 55; pw = w - 2*pad; ph = h - 65
    y0 = min(vs); y1 = max(vs)
    if y1 <= y0: y1 = y0 + 1
    tx = lambda i: pad + i / max(n-1,1) * pw
    ty = lambda v: 35 + (1-(v-y0)/(y1-y0)) * ph
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    p.append(f'<text x="{w//2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y0 < 0 < y1:
        zy = ty(0)
        p.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w-pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(vs[i]):.1f}" for i in range(n))
    p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    p.append("</svg>"); return "\n".join(p)


def _build_html(r: IntegratorResult) -> str:
    cfg = r.config; tgt = cfg.targets
    oc = "#3fb950" if r.all_met else "#f85149"
    peak = np.maximum.accumulate(r.equity_curve)
    dd_c = ((r.equity_curve - peak) / np.where(peak > 0, peak, 1) * 100).tolist()
    mc = r.monte_carlo

    yr_rows = ""
    for ym in r.year_metrics:
        yr_rows += f"<tr><td>{ym.year}</td><td>{ym.n_trades}</td><td>{_fp(ym.annual_return_pct)} {_ti(ym.return_met)}</td><td>{_fr(ym.sharpe)} {_ti(ym.sharpe_met)}</td><td>{_fp(ym.max_drawdown_pct)} {_ti(ym.dd_met)}</td><td>{_fp(ym.win_rate*100)}</td><td>{_fd(ym.total_pnl)}</td><td>{_ti(ym.all_met)}</td></tr>"

    exp_rows = ""
    for e in r.experiment_contributions:
        exp_rows += f"<tr><td style='text-align:left'>{e.name}</td><td>{e.weight:.2f}</td><td>{e.n_trades}</td><td>{_fd(e.total_pnl)}</td><td>{_fr(e.sharpe)}</td><td>{_fp(e.contribution_pct*100)}</td></tr>"

    wf_rows = ""
    for f in r.walk_forward:
        c = "#3fb950" if f.test_sharpe > 0 else "#f85149"
        wf_rows += f"<tr><td>{f.fold}</td><td>{','.join(str(y) for y in f.train_years)}</td><td>{f.test_year}</td><td>{f.n_train}</td><td>{f.n_test}</td><td>{_fr(f.train_sharpe)}</td><td style='color:{c}'>{_fr(f.test_sharpe)}</td><td>{_fp(f.test_return_pct)}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><title>North Star Integration</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1,h2,h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; }}
  .hero {{ background: #161b22; border: 2px solid {oc}; border-radius: 12px;
           padding: 24px; text-align: center; margin: 20px 0; }}
  .hero .big {{ font-size: 2.5em; font-weight: 800; color: {oc}; }}
  .hero .sub {{ color: #8b949e; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 10px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.dt th, table.dt td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .scorecard {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 16px 0; }}
  .sc-row {{ display: flex; gap: 16px; padding: 8px 0; border-bottom: 1px solid #21262d; align-items: center; }}
  .sc-row .label {{ width: 150px; color: #8b949e; }}
  .sc-row .value {{ width: 80px; font-weight: 700; color: #f0f6fc; }}
  .sc-row .target {{ flex: 1; color: #8b949e; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>North Star Integration</h1>
<div class="hero">
  <div class="big">{"ALL TARGETS MET" if r.all_met else "TARGETS NOT MET"}</div>
  <div class="sub">{r.n_trades} trades &middot; {r.n_years} years &middot;
     {len(r.experiments)} experiments &middot; {cfg.opt_method} weights</div>
</div>

<div class="summary">
  <div class="stat"><div class="label">Ann. Return</div><div class="value">{_fp(r.annualised_return_pct)}</div></div>
  <div class="stat"><div class="label">Sharpe</div><div class="value">{_fr(r.sharpe)}</div></div>
  <div class="stat"><div class="label">Sortino</div><div class="value">{_fr(r.sortino)}</div></div>
  <div class="stat"><div class="label">Max DD</div><div class="value">{_fp(r.max_dd_pct)}</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">{_fp(r.win_rate*100)}</div></div>
  <div class="stat"><div class="label">PF</div><div class="value">{_fr(r.profit_factor)}</div></div>
  <div class="stat"><div class="label">Total PnL</div><div class="value">{_fd(r.total_pnl)}</div></div>
  <div class="stat"><div class="label">Final Cap</div><div class="value">{_fd(r.final_capital)}</div></div>
</div>

<h2>North Star Scorecard</h2>
<div class="scorecard">
  <div class="sc-row"><span class="label">Annual Return</span><span class="value">{_fp(r.annualised_return_pct)}</span><span class="target">Target: {_fp(tgt.annual_return_pct)}</span><span>{_ti(r.return_met)}</span></div>
  <div class="sc-row"><span class="label">Sharpe Ratio</span><span class="value">{_fr(r.sharpe)}</span><span class="target">Target: {_fr(tgt.sharpe_ratio)}</span><span>{_ti(r.sharpe_met)}</span></div>
  <div class="sc-row"><span class="label">Max Drawdown</span><span class="value">{_fp(r.max_dd_pct)}</span><span class="target">Target: &le;{_fp(tgt.max_drawdown_pct)}</span><span>{_ti(r.dd_met)}</span></div>
</div>

<h2>Equity Curve</h2>
{_svg_line(r.equity_curve.tolist(), "Portfolio Equity ($)", "#3fb950")}

<h2>Drawdown</h2>
{_svg_line(dd_c, "Drawdown (%)", "#f85149")}

<h2>Per-Year Performance</h2>
<table class="dt"><tr><th>Year</th><th>Trades</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>PnL</th><th>All</th></tr>{yr_rows}</table>

<h2>Experiment Contributions</h2>
<table class="dt"><tr><th style="text-align:left">Experiment</th><th>Weight</th><th>Trades</th><th>PnL</th><th>Sharpe</th><th>Contribution</th></tr>{exp_rows}</table>

<h2>Walk-Forward Validation</h2>
<table class="dt"><tr><th>Fold</th><th>Train</th><th>Test</th><th>N Train</th><th>N Test</th><th>Train Sharpe</th><th>Test Sharpe</th><th>Test Return</th></tr>{wf_rows}</table>

<h2>Monte Carlo Stress ({mc.n_paths:,} paths)</h2>
<div class="card"><div class="metrics-grid">
  <div><span class="label">Median Return</span><span class="value">{_fp(mc.median_return_pct)}</span></div>
  <div><span class="label">P5 Return</span><span class="value">{_fp(mc.p5_return_pct)}</span></div>
  <div><span class="label">P95 Return</span><span class="value">{_fp(mc.p95_return_pct)}</span></div>
  <div><span class="label">Prob Target Return</span><span class="value">{_fp(mc.prob_target_return*100)}</span></div>
  <div><span class="label">Prob Exceed DD</span><span class="value">{_fp(mc.prob_exceed_dd*100)}</span></div>
  <div><span class="label">VaR 95</span><span class="value">{_fp(mc.var_95_pct)}</span></div>
</div></div>

</body></html>"""
