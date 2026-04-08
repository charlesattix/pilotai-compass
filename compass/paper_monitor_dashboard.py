"""Paper trading monitoring dashboard — unified HTML view for EXP-880
standalone and EXP-1470 combined portfolio with P&L tracking, Greeks,
drawdown alerts, signal quality, and regime state.

Designed for daily/weekly email-ready HTML summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── Backtest expectations ───────────────────────────────────────────────────
EXPECTATIONS = {
    "EXP-880": {"cagr": 76.9, "sharpe": 4.97, "max_dd": 10.2, "win_rate": 75.0},
    "EXP-1470": {"cagr": 95.0, "sharpe": 6.0, "max_dd": 8.0, "win_rate": 78.0},
}

DD_WARN = 5.0
DD_CRIT = 10.0
DD_HALT = 13.0


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class PositionInfo:
    symbol: str
    qty: int
    side: str
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    pnl: float = 0.0


@dataclass
class SignalQuality:
    total_signals: int = 0
    correct_direction: int = 0
    accuracy_pct: float = 0.0
    avg_confidence: float = 0.0
    avg_predicted_pnl: float = 0.0
    avg_actual_pnl: float = 0.0
    calibration_error: float = 0.0  # |predicted - actual| / |actual|


@dataclass
class RegimeState:
    current_regime: str = "bull"
    hedge_scale: float = 1.0
    leverage: float = 2.0
    vix: float = 18.0
    dd_controller_active: bool = False


@dataclass
class StrategyView:
    strategy_id: str
    equity: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    current_dd_pct: float = 0.0
    sharpe: float = 0.0
    win_rate_pct: float = 0.0
    n_trades: int = 0
    positions: List[PositionInfo] = field(default_factory=list)
    signal_quality: Optional[SignalQuality] = None
    regime: Optional[RegimeState] = None
    # Deviation from backtest
    expected_cagr: float = 0.0
    cagr_deviation_pct: float = 0.0
    alerts: List[str] = field(default_factory=list)


@dataclass
class DashboardResult:
    views: List[StrategyView] = field(default_factory=list)
    combined_equity: float = 0.0
    combined_pnl: float = 0.0
    combined_dd_pct: float = 0.0
    combined_sharpe: float = 0.0
    portfolio_delta: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0
    alerts: List[str] = field(default_factory=list)
    period: str = "daily"
    generated_at: str = ""


# ── Core dashboard ──────────────────────────────────────────────────────────
class PaperMonitorDashboard:
    """Generates monitoring dashboards for paper trading strategies."""

    def __init__(self, starting_capital: float = 100_000.0) -> None:
        self.starting_capital = starting_capital

    def build(
        self,
        strategy_data: Dict[str, Dict[str, Any]],
        period: str = "daily",
    ) -> DashboardResult:
        """Build dashboard from strategy data.

        strategy_data: strategy_id → {
            equity, total_pnl, current_dd_pct, sharpe, win_rate_pct,
            n_trades, positions: [{symbol, qty, side, delta, ...}],
            signals: {total, correct, avg_confidence, avg_predicted, avg_actual},
            regime: {current, hedge_scale, leverage, vix, dd_active},
        }
        """
        views: List[StrategyView] = []
        all_alerts: List[str] = []
        total_delta = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for sid, data in strategy_data.items():
            view = self._build_view(sid, data)
            views.append(view)
            all_alerts.extend(view.alerts)

            for p in view.positions:
                total_delta += p.delta
                total_theta += p.theta
                total_vega += p.vega

        # Combined metrics
        equities = [v.equity for v in views if v.equity > 0]
        combined_eq = sum(equities)
        combined_pnl = sum(v.total_pnl for v in views)
        combined_dd = max((v.current_dd_pct for v in views), default=0)

        # Combined Sharpe (simple average)
        sharpes = [v.sharpe for v in views if v.sharpe != 0]
        combined_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

        # Portfolio-level DD alerts
        if combined_dd >= DD_HALT:
            all_alerts.insert(0, f"HALT: Combined DD {combined_dd:.1f}% ≥ {DD_HALT}%")
        elif combined_dd >= DD_CRIT:
            all_alerts.insert(0, f"CRITICAL: Combined DD {combined_dd:.1f}%")
        elif combined_dd >= DD_WARN:
            all_alerts.insert(0, f"WARNING: Combined DD {combined_dd:.1f}%")

        return DashboardResult(
            views=views,
            combined_equity=round(combined_eq, 2),
            combined_pnl=round(combined_pnl, 2),
            combined_dd_pct=round(combined_dd, 2),
            combined_sharpe=round(combined_sharpe, 2),
            portfolio_delta=round(total_delta, 2),
            portfolio_theta=round(total_theta, 2),
            portfolio_vega=round(total_vega, 2),
            alerts=all_alerts,
            period=period,
            generated_at=_now(),
        )

    def _build_view(self, sid: str, data: Dict) -> StrategyView:
        equity = float(data.get("equity", self.starting_capital))
        total_pnl = float(data.get("total_pnl", 0))
        ret_pct = (equity - self.starting_capital) / self.starting_capital * 100
        dd = float(data.get("current_dd_pct", 0))
        sharpe = float(data.get("sharpe", 0))
        wr = float(data.get("win_rate_pct", 0))
        n_trades = int(data.get("n_trades", 0))

        # Positions
        positions = []
        for p in data.get("positions", []):
            positions.append(PositionInfo(
                symbol=p.get("symbol", ""),
                qty=int(p.get("qty", 0)),
                side=p.get("side", ""),
                delta=float(p.get("delta", 0)),
                gamma=float(p.get("gamma", 0)),
                theta=float(p.get("theta", 0)),
                vega=float(p.get("vega", 0)),
                pnl=float(p.get("pnl", 0)),
            ))

        # Signal quality
        sq = None
        sig_data = data.get("signals")
        if sig_data:
            total = int(sig_data.get("total", 0))
            correct = int(sig_data.get("correct", 0))
            acc = correct / total * 100 if total > 0 else 0
            avg_pred = float(sig_data.get("avg_predicted", 0))
            avg_act = float(sig_data.get("avg_actual", 0))
            cal_err = abs(avg_pred - avg_act) / max(abs(avg_act), 0.01) if avg_act != 0 else 0
            sq = SignalQuality(
                total_signals=total, correct_direction=correct,
                accuracy_pct=round(acc, 1),
                avg_confidence=float(sig_data.get("avg_confidence", 0)),
                avg_predicted_pnl=round(avg_pred, 2),
                avg_actual_pnl=round(avg_act, 2),
                calibration_error=round(cal_err, 4),
            )

        # Regime
        regime = None
        reg_data = data.get("regime")
        if reg_data:
            regime = RegimeState(
                current_regime=reg_data.get("current", "bull"),
                hedge_scale=float(reg_data.get("hedge_scale", 1.0)),
                leverage=float(reg_data.get("leverage", 2.0)),
                vix=float(reg_data.get("vix", 18.0)),
                dd_controller_active=bool(reg_data.get("dd_active", False)),
            )

        # Deviation from expectations
        exp = EXPECTATIONS.get(sid, {})
        expected_cagr = exp.get("cagr", 0)
        cagr_dev = ((ret_pct - expected_cagr) / max(abs(expected_cagr), 0.01) * 100) if expected_cagr else 0

        # Alerts
        alerts: List[str] = []
        if dd >= DD_HALT:
            alerts.append(f"{sid}: HALT — DD {dd:.1f}%")
        elif dd >= DD_CRIT:
            alerts.append(f"{sid}: CRITICAL — DD {dd:.1f}%")
        elif dd >= DD_WARN:
            alerts.append(f"{sid}: WARNING — DD {dd:.1f}%")

        if sq and sq.accuracy_pct < 50:
            alerts.append(f"{sid}: Signal accuracy {sq.accuracy_pct:.0f}% < 50%")
        if sq and sq.calibration_error > 0.5:
            alerts.append(f"{sid}: Calibration error {sq.calibration_error:.1%}")

        if regime and regime.hedge_scale < 0.50:
            alerts.append(f"{sid}: Hedge active (scale {regime.hedge_scale:.0%})")

        return StrategyView(
            strategy_id=sid,
            equity=round(equity, 2),
            total_pnl=round(total_pnl, 2),
            return_pct=round(ret_pct, 2),
            current_dd_pct=round(dd, 2),
            sharpe=round(sharpe, 2),
            win_rate_pct=round(wr, 1),
            n_trades=n_trades,
            positions=positions,
            signal_quality=sq,
            regime=regime,
            expected_cagr=expected_cagr,
            cagr_deviation_pct=round(cagr_dev, 1),
            alerts=alerts,
        )

    # ── HTML report ─────────────────────────────────────────────────────────
    def generate_report(
        self, result: DashboardResult, output_path: str = "reports/paper_monitor.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._build_html(result), encoding="utf-8")
        return path

    def _build_html(self, r: DashboardResult) -> str:
        cards = self._html_cards(r)
        alerts = self._html_alerts(r.alerts)
        strat_cards = "".join(self._html_strategy(v) for v in r.views)
        greeks = self._html_greeks(r)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Paper Trading Monitor — {r.period.title()}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#fff;color:#1e293b;padding:20px;max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1rem;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:3px;margin:16px 0 8px}}
h3{{font-size:.9rem;color:#475569;margin:10px 0 6px}}
.sub{{color:#64748b;font-size:.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px}}
.card .lbl{{font-size:.68rem;color:#64748b;text-transform:uppercase}}
.card .val{{font-size:1.15rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.8rem;margin-bottom:10px}}
th,td{{padding:5px 8px;text-align:left;border-bottom:1px solid #e2e8f0}}
th{{color:#64748b;background:#f8fafc;font-size:.75rem}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}.warn{{color:#d97706}}
.alert{{padding:8px 12px;border-radius:5px;margin-bottom:6px;font-size:.82rem}}
.alert.crit{{background:#fef2f2;border-left:3px solid #dc2626;color:#991b1b}}
.alert.warn{{background:#fffbeb;border-left:3px solid #d97706;color:#92400e}}
.alert.info{{background:#f0f9ff;border-left:3px solid #0284c7;color:#075985}}
.strat-card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-bottom:14px}}
</style>
</head>
<body>
<h1>Paper Trading Monitor</h1>
<p class="sub">{r.period.title()} Report — {r.generated_at}</p>
{alerts}{cards}{greeks}
<h2>Per-Strategy Views</h2>
{strat_cards}
<p style="color:#94a3b8;font-size:.7rem;margin-top:16px">Generated by compass/paper_monitor_dashboard.py</p>
</body></html>"""

    @staticmethod
    def _html_cards(r: DashboardResult) -> str:
        dd_cls = "neg" if r.combined_dd_pct >= DD_CRIT else "warn" if r.combined_dd_pct >= DD_WARN else ""
        return f"""<div class="grid">
<div class="card"><div class="lbl">Combined Equity</div><div class="val">${r.combined_equity:,.0f}</div></div>
<div class="card"><div class="lbl">Total P&L</div><div class="val {'pos' if r.combined_pnl>=0 else 'neg'}">${r.combined_pnl:+,.0f}</div></div>
<div class="card"><div class="lbl">Drawdown</div><div class="val {dd_cls}">{r.combined_dd_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Sharpe</div><div class="val">{r.combined_sharpe:.2f}</div></div>
<div class="card"><div class="lbl">Strategies</div><div class="val">{len(r.views)}</div></div>
<div class="card"><div class="lbl">Alerts</div><div class="val {'neg' if r.alerts else ''}">{len(r.alerts)}</div></div>
</div>"""

    @staticmethod
    def _html_alerts(alerts: List[str]) -> str:
        if not alerts:
            return '<div class="alert info">No active alerts — system nominal</div>'
        items = ""
        for a in alerts:
            cls = "crit" if "HALT" in a or "CRITICAL" in a else "warn"
            items += f'<div class="alert {cls}">{a}</div>'
        return items

    @staticmethod
    def _html_greeks(r: DashboardResult) -> str:
        return f"""<h2>Portfolio Greeks</h2>
<div class="grid">
<div class="card"><div class="lbl">Delta</div><div class="val">{r.portfolio_delta:.1f}</div></div>
<div class="card"><div class="lbl">Theta</div><div class="val">{r.portfolio_theta:.1f}/day</div></div>
<div class="card"><div class="lbl">Vega</div><div class="val">{r.portfolio_vega:.1f}</div></div>
</div>"""

    def _html_strategy(self, v: StrategyView) -> str:
        dd_cls = "neg" if v.current_dd_pct >= DD_CRIT else "warn" if v.current_dd_pct >= DD_WARN else ""
        dev_cls = "neg" if v.cagr_deviation_pct < -30 else "warn" if v.cagr_deviation_pct < -10 else "pos"

        pos_rows = ""
        for p in v.positions[:10]:
            pos_rows += (f"<tr><td>{p.symbol}</td><td>{p.qty}</td><td>{p.side}</td>"
                         f"<td>{p.delta:.2f}</td><td>{p.theta:.2f}</td>"
                         f'<td class="{"pos" if p.pnl>=0 else "neg"}">${p.pnl:+,.0f}</td></tr>')

        sig_html = ""
        if v.signal_quality:
            sq = v.signal_quality
            sig_html = f"""<h3>Signal Quality</h3>
<table><tbody>
<tr><td>Accuracy</td><td>{sq.accuracy_pct:.0f}%</td><td>({sq.correct_direction}/{sq.total_signals})</td></tr>
<tr><td>Avg Confidence</td><td>{sq.avg_confidence:.2f}</td><td></td></tr>
<tr><td>Predicted P&L</td><td>${sq.avg_predicted_pnl:,.0f}</td><td></td></tr>
<tr><td>Actual P&L</td><td>${sq.avg_actual_pnl:,.0f}</td><td></td></tr>
<tr><td>Calibration Error</td><td>{sq.calibration_error:.1%}</td><td></td></tr>
</tbody></table>"""

        regime_html = ""
        if v.regime:
            rg = v.regime
            regime_html = f"""<h3>Regime & Leverage</h3>
<table><tbody>
<tr><td>Regime</td><td>{rg.current_regime}</td></tr>
<tr><td>Hedge Scale</td><td class="{'warn' if rg.hedge_scale<0.8 else ''}">{rg.hedge_scale:.0%}</td></tr>
<tr><td>Leverage</td><td>{rg.leverage:.1f}x</td></tr>
<tr><td>VIX</td><td>{rg.vix:.1f}</td></tr>
<tr><td>DD Controller</td><td>{'ACTIVE' if rg.dd_controller_active else 'Off'}</td></tr>
</tbody></table>"""

        return f"""<div class="strat-card">
<h2>{v.strategy_id}</h2>
<div class="grid">
<div class="card"><div class="lbl">Equity</div><div class="val">${v.equity:,.0f}</div></div>
<div class="card"><div class="lbl">Return</div><div class="val {'pos' if v.return_pct>=0 else 'neg'}">{v.return_pct:+.1f}%</div></div>
<div class="card"><div class="lbl">DD</div><div class="val {dd_cls}">{v.current_dd_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Sharpe</div><div class="val">{v.sharpe:.2f}</div></div>
<div class="card"><div class="lbl">Win Rate</div><div class="val">{v.win_rate_pct:.0f}%</div></div>
<div class="card"><div class="lbl">vs Backtest</div><div class="val {dev_cls}">{v.cagr_deviation_pct:+.0f}%</div></div>
</div>
{'<h3>Positions</h3><table><thead><tr><th>Symbol</th><th>Qty</th><th>Side</th><th>Delta</th><th>Theta</th><th>P&L</th></tr></thead><tbody>' + pos_rows + '</tbody></table>' if pos_rows else ''}
{sig_html}{regime_html}
</div>"""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
