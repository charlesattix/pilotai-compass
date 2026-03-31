"""
compass/portfolio_dashboard.py — Master portfolio dashboard.

Aggregates experiment metrics, risk budgets, signal quality, recent trades,
alerts, and regime information into a single investor-quality HTML report
with inline SVG charts.  No external JS/CSS dependencies.

Usage::

    from compass.portfolio_dashboard import PortfolioDashboard

    dash = PortfolioDashboard()
    result = dash.build(
        experiment_metrics={...},
        risk_budget={...},
        signal_quality={...},
        recent_trades=[...],
        alerts=[...],
        regime_info={...},
    )
    dash.generate_report(result, "reports/portfolio_dashboard.html")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Traffic-light constants ─────────────────────────────────────────────────
GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

# ── Severity ordering for alerts ────────────────────────────────────────────
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# ── Default Sharpe thresholds for experiment status ─────────────────────────
DEFAULT_SHARPE_GREEN = 1.5
DEFAULT_SHARPE_YELLOW = 0.8


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ExecutiveSummary:
    """Portfolio-level headline numbers."""
    portfolio_sharpe: float = 0.0
    total_return_pct: float = 0.0
    worst_drawdown_pct: float = 0.0
    estimated_capacity: float = 0.0
    n_experiments: int = 0
    n_green: int = 0
    n_yellow: int = 0
    n_red: int = 0
    overall_status: str = GREEN


@dataclass
class ExperimentCard:
    """Per-experiment status card."""
    experiment_id: str
    status: str  # GREEN / YELLOW / RED
    sharpe: float = 0.0
    return_pct: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    capacity: float = 0.0


@dataclass
class RegimePanel:
    """Regime regime information."""
    current: str = "unknown"
    forecast: str = "unknown"
    confidence: float = 0.0


@dataclass
class RiskBudgetRow:
    """Single experiment's risk budget contribution."""
    experiment_id: str
    var: float = 0.0
    cvar: float = 0.0
    contribution_pct: float = 0.0


@dataclass
class SignalQualityRow:
    """Signal quality metrics for one experiment."""
    experiment_id: str
    ic: float = 0.0
    decay_hours: float = 0.0
    snr: float = 0.0


@dataclass
class TradeLogEntry:
    """Single trade in the recent trade log."""
    date: str = ""
    pnl: float = 0.0
    experiment: str = ""
    trade_type: str = ""


@dataclass
class AlertEntry:
    """Single alert / anomaly."""
    severity: str = "info"
    message: str = ""
    timestamp: str = ""


@dataclass
class DashboardResult:
    """Complete dashboard output — every section is a typed field."""
    executive_summary: Optional[ExecutiveSummary] = None
    experiment_cards: List[ExperimentCard] = field(default_factory=list)
    regime_panel: Optional[RegimePanel] = None
    risk_budget: List[RiskBudgetRow] = field(default_factory=list)
    signal_quality: List[SignalQualityRow] = field(default_factory=list)
    trade_log: List[TradeLogEntry] = field(default_factory=list)
    alerts: List[AlertEntry] = field(default_factory=list)
    generated_at: str = ""


# ── Core dashboard ──────────────────────────────────────────────────────────
class PortfolioDashboard:
    """Master portfolio dashboard aggregator."""

    def __init__(
        self,
        sharpe_green: float = DEFAULT_SHARPE_GREEN,
        sharpe_yellow: float = DEFAULT_SHARPE_YELLOW,
        max_recent_trades: int = 20,
    ) -> None:
        self.sharpe_green = sharpe_green
        self.sharpe_yellow = sharpe_yellow
        self.max_recent_trades = max_recent_trades

    # ── Public API ──────────────────────────────────────────────────────────
    def build(
        self,
        experiment_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        risk_budget: Optional[Dict[str, Dict[str, Any]]] = None,
        signal_quality: Optional[Dict[str, Dict[str, Any]]] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> DashboardResult:
        """Build all dashboard sections from raw dict inputs.

        Parameters
        ----------
        experiment_metrics : dict[str, dict]
            experiment_id -> {sharpe, return_pct, max_dd, win_rate,
            profit_factor, total_trades, capacity, ...}
        risk_budget : dict[str, dict]
            experiment_id -> {var, cvar, contribution_pct}
        signal_quality : dict[str, dict]
            experiment_id -> {ic, decay_hours, snr}
        recent_trades : list[dict]
            [{date, pnl, experiment, type}, ...]
        alerts : list[dict]
            [{severity, message, timestamp}, ...]
        regime_info : dict
            {current, forecast, confidence}
        """
        experiment_metrics = experiment_metrics or {}
        risk_budget = risk_budget or {}
        signal_quality = signal_quality or {}
        recent_trades = recent_trades or []
        alerts = alerts or []
        regime_info = regime_info or {}

        cards = self._build_experiment_cards(experiment_metrics)
        summary = self._build_executive_summary(experiment_metrics, cards)
        regime = self._build_regime_panel(regime_info)
        rb_rows = self._build_risk_budget(risk_budget)
        sq_rows = self._build_signal_quality(signal_quality)
        trade_log = self._build_trade_log(recent_trades)
        alert_list = self._build_alerts(alerts)

        return DashboardResult(
            executive_summary=summary,
            experiment_cards=cards,
            regime_panel=regime,
            risk_budget=rb_rows,
            signal_quality=sq_rows,
            trade_log=trade_log,
            alerts=alert_list,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: DashboardResult,
        output_path: str | Path = "reports/portfolio_dashboard.html",
    ) -> Path:
        """Write self-contained HTML dashboard with inline SVG charts."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Portfolio dashboard written to %s", path)
        return path

    # ── Section builders ────────────────────────────────────────────────────
    def _status_for_sharpe(self, sharpe: float) -> str:
        if sharpe >= self.sharpe_green:
            return GREEN
        elif sharpe >= self.sharpe_yellow:
            return YELLOW
        return RED

    def _build_experiment_cards(
        self, metrics: Dict[str, Dict[str, Any]],
    ) -> List[ExperimentCard]:
        cards: List[ExperimentCard] = []
        for eid, m in sorted(metrics.items()):
            sharpe = float(m.get("sharpe", 0.0))
            cards.append(ExperimentCard(
                experiment_id=eid,
                status=self._status_for_sharpe(sharpe),
                sharpe=sharpe,
                return_pct=float(m.get("return_pct", 0.0)),
                max_dd=float(m.get("max_dd", 0.0)),
                win_rate=float(m.get("win_rate", 0.0)),
                profit_factor=float(m.get("profit_factor", 0.0)),
                total_trades=int(m.get("total_trades", 0)),
                capacity=float(m.get("capacity", 0.0)),
            ))
        return cards

    def _build_executive_summary(
        self,
        metrics: Dict[str, Dict[str, Any]],
        cards: List[ExperimentCard],
    ) -> ExecutiveSummary:
        if not metrics:
            return ExecutiveSummary()

        sharpes = [float(m.get("sharpe", 0.0)) for m in metrics.values()]
        returns = [float(m.get("return_pct", 0.0)) for m in metrics.values()]
        dds = [float(m.get("max_dd", 0.0)) for m in metrics.values()]
        caps = [float(m.get("capacity", 0.0)) for m in metrics.values()]

        # Portfolio Sharpe: equal-weight average (simple aggregation)
        portfolio_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        total_return = float(np.mean(returns)) if returns else 0.0
        worst_dd = float(np.min(dds)) if dds else 0.0  # most negative
        estimated_cap = float(np.sum(caps)) if caps else 0.0

        n_green = sum(1 for c in cards if c.status == GREEN)
        n_yellow = sum(1 for c in cards if c.status == YELLOW)
        n_red = sum(1 for c in cards if c.status == RED)

        if n_red > 0:
            overall = RED
        elif n_yellow > 0:
            overall = YELLOW
        else:
            overall = GREEN

        return ExecutiveSummary(
            portfolio_sharpe=portfolio_sharpe,
            total_return_pct=total_return,
            worst_drawdown_pct=worst_dd,
            estimated_capacity=estimated_cap,
            n_experiments=len(cards),
            n_green=n_green,
            n_yellow=n_yellow,
            n_red=n_red,
            overall_status=overall,
        )

    @staticmethod
    def _build_regime_panel(info: Dict[str, Any]) -> RegimePanel:
        return RegimePanel(
            current=str(info.get("current", "unknown")),
            forecast=str(info.get("forecast", "unknown")),
            confidence=float(info.get("confidence", 0.0)),
        )

    @staticmethod
    def _build_risk_budget(
        rb: Dict[str, Dict[str, Any]],
    ) -> List[RiskBudgetRow]:
        rows: List[RiskBudgetRow] = []
        for eid, data in sorted(rb.items()):
            rows.append(RiskBudgetRow(
                experiment_id=eid,
                var=float(data.get("var", 0.0)),
                cvar=float(data.get("cvar", 0.0)),
                contribution_pct=float(data.get("contribution_pct", 0.0)),
            ))
        return rows

    @staticmethod
    def _build_signal_quality(
        sq: Dict[str, Dict[str, Any]],
    ) -> List[SignalQualityRow]:
        rows: List[SignalQualityRow] = []
        for eid, data in sorted(sq.items()):
            rows.append(SignalQualityRow(
                experiment_id=eid,
                ic=float(data.get("ic", 0.0)),
                decay_hours=float(data.get("decay_hours", 0.0)),
                snr=float(data.get("snr", 0.0)),
            ))
        return rows

    def _build_trade_log(
        self, trades: List[Dict[str, Any]],
    ) -> List[TradeLogEntry]:
        entries: List[TradeLogEntry] = []
        for t in trades[: self.max_recent_trades]:
            entries.append(TradeLogEntry(
                date=str(t.get("date", "")),
                pnl=float(t.get("pnl", 0.0)),
                experiment=str(t.get("experiment", "")),
                trade_type=str(t.get("type", "")),
            ))
        return entries

    @staticmethod
    def _build_alerts(
        raw_alerts: List[Dict[str, Any]],
    ) -> List[AlertEntry]:
        parsed: List[AlertEntry] = []
        for a in raw_alerts:
            parsed.append(AlertEntry(
                severity=str(a.get("severity", "info")).lower(),
                message=str(a.get("message", "")),
                timestamp=str(a.get("timestamp", "")),
            ))
        # Sort by severity: critical first
        parsed.sort(key=lambda x: SEVERITY_ORDER.get(x.severity, 99))
        return parsed

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── HTML generation ─────────────────────────────────────────────────────
    def _build_html(self, r: DashboardResult) -> str:
        summary_html = self._html_executive_summary(r.executive_summary)
        cards_html = self._html_experiment_cards(r.experiment_cards)
        regime_html = self._html_regime_panel(r.regime_panel)
        risk_html = self._html_risk_budget(r.risk_budget)
        signal_html = self._html_signal_quality(r.signal_quality)
        trades_html = self._html_trade_log(r.trade_log)
        alerts_html = self._html_alerts(r.alerts)
        charts_html = self._html_charts(r)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Portfolio Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:1.7rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.15rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600;background:#1e293b}}
tr:hover{{background:#1e293b88}}
.GREEN{{color:#4ade80}}.YELLOW{{color:#fbbf24}}.RED{{color:#f87171}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.dot.GREEN{{background:#4ade80}}.dot.YELLOW{{background:#fbbf24}}.dot.RED{{background:#f87171}}
.exp-card{{background:#1e293b;border-radius:10px;padding:16px;border-left:4px solid #475569}}
.exp-card.GREEN{{border-color:#4ade80}}.exp-card.YELLOW{{border-color:#fbbf24}}.exp-card.RED{{border-color:#f87171}}
.exp-card h3{{font-size:.95rem;margin-bottom:8px}}
.exp-card .metrics{{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:.8rem}}
.exp-card .metrics .k{{color:#94a3b8}}.exp-card .metrics .v{{font-weight:600}}
.exp-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:28px}}
.regime-panel{{background:#1e293b;border-radius:10px;padding:20px;margin-bottom:28px;display:flex;gap:32px;align-items:center;flex-wrap:wrap}}
.regime-panel .regime-item{{text-align:center}}
.regime-panel .regime-label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase}}
.regime-panel .regime-value{{font-size:1.2rem;font-weight:700;margin-top:4px}}
.alert-critical{{color:#f87171;font-weight:700}}
.alert-high{{color:#fb923c;font-weight:600}}
.alert-medium{{color:#fbbf24}}
.alert-low{{color:#94a3b8}}
.alert-info{{color:#64748b}}
.chart-row{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:28px}}
.chart-box{{background:#1e293b;border-radius:10px;padding:16px;flex:1;min-width:300px}}
.chart-box h3{{font-size:.9rem;color:#38bdf8;margin-bottom:8px}}
svg text{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
</style>
</head>
<body>
<h1>Portfolio Dashboard</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{summary_html}
{regime_html}
{charts_html}

<div class="sec">
<h2>Experiment Status Cards</h2>
<div class="exp-grid">{cards_html}</div>
</div>

{risk_html}
{signal_html}
{trades_html}
{alerts_html}

</body>
</html>"""

    # ── Executive summary cards ─────────────────────────────────────────────
    @staticmethod
    def _html_executive_summary(s: Optional[ExecutiveSummary]) -> str:
        if not s:
            return ""
        return f"""<div class="grid">
<div class="card"><div class="lbl">Overall Status</div><div class="val {s.overall_status}"><span class="dot {s.overall_status}"></span>{s.overall_status}</div></div>
<div class="card"><div class="lbl">Portfolio Sharpe</div><div class="val">{s.portfolio_sharpe:.2f}</div></div>
<div class="card"><div class="lbl">Total Return</div><div class="val">{s.total_return_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Worst Drawdown</div><div class="val">{s.worst_drawdown_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Est. Capacity</div><div class="val">${s.estimated_capacity:,.0f}</div></div>
<div class="card"><div class="lbl">Experiments</div><div class="val">{s.n_experiments}</div></div>
<div class="card"><div class="lbl">Traffic Lights</div><div class="val"><span class="GREEN">{s.n_green}G</span> <span class="YELLOW">{s.n_yellow}Y</span> <span class="RED">{s.n_red}R</span></div></div>
</div>"""

    # ── Experiment cards ────────────────────────────────────────────────────
    @staticmethod
    def _html_experiment_cards(cards: List[ExperimentCard]) -> str:
        if not cards:
            return "<p>No experiments.</p>"
        out = ""
        for c in cards:
            out += f"""<div class="exp-card {c.status}">
<h3><span class="dot {c.status}"></span>{c.experiment_id}</h3>
<div class="metrics">
<span class="k">Sharpe</span><span class="v">{c.sharpe:.2f}</span>
<span class="k">Return</span><span class="v">{c.return_pct:.1f}%</span>
<span class="k">Max DD</span><span class="v">{c.max_dd:.1f}%</span>
<span class="k">Win Rate</span><span class="v">{c.win_rate:.1f}%</span>
<span class="k">PF</span><span class="v">{c.profit_factor:.2f}</span>
<span class="k">Trades</span><span class="v">{c.total_trades}</span>
<span class="k">Capacity</span><span class="v">${c.capacity:,.0f}</span>
</div>
</div>"""
        return out

    # ── Regime panel ────────────────────────────────────────────────────────
    @staticmethod
    def _html_regime_panel(rp: Optional[RegimePanel]) -> str:
        if not rp:
            return ""
        conf_pct = rp.confidence * 100 if rp.confidence <= 1.0 else rp.confidence
        return f"""<div class="regime-panel">
<div class="regime-item"><div class="regime-label">Current Regime</div><div class="regime-value">{rp.current}</div></div>
<div class="regime-item"><div class="regime-label">Forecast</div><div class="regime-value">{rp.forecast}</div></div>
<div class="regime-item"><div class="regime-label">Confidence</div><div class="regime-value">{conf_pct:.0f}%</div></div>
</div>"""

    # ── Risk budget table ───────────────────────────────────────────────────
    @staticmethod
    def _html_risk_budget(rows: List[RiskBudgetRow]) -> str:
        if not rows:
            return ""
        tbody = ""
        for r in rows:
            tbody += (
                f"<tr><td>{r.experiment_id}</td>"
                f"<td>{r.var:.4f}</td>"
                f"<td>{r.cvar:.4f}</td>"
                f"<td>{r.contribution_pct:.1f}%</td></tr>"
            )
        return f"""<div class="sec">
<h2>Risk Budget</h2>
<table>
<thead><tr><th>Experiment</th><th>VaR</th><th>CVaR</th><th>Contribution %</th></tr></thead>
<tbody>{tbody}</tbody>
</table>
</div>"""

    # ── Signal quality table ────────────────────────────────────────────────
    @staticmethod
    def _html_signal_quality(rows: List[SignalQualityRow]) -> str:
        if not rows:
            return ""
        tbody = ""
        for r in rows:
            tbody += (
                f"<tr><td>{r.experiment_id}</td>"
                f"<td>{r.ic:.4f}</td>"
                f"<td>{r.decay_hours:.1f}h</td>"
                f"<td>{r.snr:.2f}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Signal Quality</h2>
<table>
<thead><tr><th>Experiment</th><th>IC</th><th>Decay Half-Life</th><th>SNR</th></tr></thead>
<tbody>{tbody}</tbody>
</table>
</div>"""

    # ── Trade log table ─────────────────────────────────────────────────────
    @staticmethod
    def _html_trade_log(entries: List[TradeLogEntry]) -> str:
        if not entries:
            return ""
        tbody = ""
        for t in entries:
            cls = "GREEN" if t.pnl >= 0 else "RED"
            tbody += (
                f"<tr><td>{t.date}</td>"
                f"<td>{t.experiment}</td>"
                f"<td>{t.trade_type}</td>"
                f'<td class="{cls}">${t.pnl:+,.2f}</td></tr>'
            )
        return f"""<div class="sec">
<h2>Recent Trades</h2>
<table>
<thead><tr><th>Date</th><th>Experiment</th><th>Type</th><th>P&amp;L</th></tr></thead>
<tbody>{tbody}</tbody>
</table>
</div>"""

    # ── Alerts panel ────────────────────────────────────────────────────────
    @staticmethod
    def _html_alerts(alerts: List[AlertEntry]) -> str:
        if not alerts:
            return ""
        tbody = ""
        for a in alerts:
            cls = f"alert-{a.severity}"
            tbody += (
                f'<tr><td class="{cls}">{a.severity.upper()}</td>'
                f"<td>{a.message}</td>"
                f"<td>{a.timestamp}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Alerts</h2>
<table>
<thead><tr><th>Severity</th><th>Message</th><th>Timestamp</th></tr></thead>
<tbody>{tbody}</tbody>
</table>
</div>"""

    # ── Inline SVG charts ───────────────────────────────────────────────────
    def _html_charts(self, r: DashboardResult) -> str:
        parts: List[str] = []

        # Risk contribution pie chart (as horizontal bar)
        if r.risk_budget:
            parts.append(self._svg_risk_bar(r.risk_budget))

        # P&L bar chart from recent trades
        if r.trade_log:
            parts.append(self._svg_pnl_bars(r.trade_log))

        # Signal quality radar-ish comparison (bar chart)
        if r.signal_quality:
            parts.append(self._svg_signal_bars(r.signal_quality))

        if not parts:
            return ""
        return '<div class="chart-row">' + "".join(parts) + "</div>"

    @staticmethod
    def _svg_risk_bar(rows: List[RiskBudgetRow]) -> str:
        """Horizontal stacked bar showing risk contribution percentages."""
        w, h = 400, 60
        bar_h = 30
        y_bar = 20

        segments: List[str] = []
        colors = ["#38bdf8", "#818cf8", "#fb923c", "#4ade80", "#f472b6",
                  "#a78bfa", "#34d399", "#fbbf24"]
        x = 0
        total = sum(r.contribution_pct for r in rows) or 1.0
        labels: List[str] = []
        for i, r in enumerate(rows):
            pct = r.contribution_pct / total * 100
            seg_w = max(pct / 100 * w, 1)
            color = colors[i % len(colors)]
            segments.append(
                f'<rect x="{x:.1f}" y="{y_bar}" width="{seg_w:.1f}" '
                f'height="{bar_h}" fill="{color}" rx="2"/>'
            )
            if pct > 8:
                tx = x + seg_w / 2
                segments.append(
                    f'<text x="{tx:.1f}" y="{y_bar + bar_h / 2 + 4}" '
                    f'text-anchor="middle" fill="#0f172a" font-size="10">'
                    f'{r.experiment_id}</text>'
                )
            labels.append(
                f'<tspan fill="{color}">{r.experiment_id} '
                f'{r.contribution_pct:.0f}%</tspan>'
            )
            x += seg_w

        label_text = "  ".join(labels)
        return f"""<div class="chart-box">
<h3>Risk Contribution</h3>
<svg viewBox="0 0 {w} {h + 20}" width="100%" xmlns="http://www.w3.org/2000/svg">
<text x="0" y="14" fill="#94a3b8" font-size="10">{label_text}</text>
{"".join(segments)}
</svg>
</div>"""

    @staticmethod
    def _svg_pnl_bars(entries: List[TradeLogEntry]) -> str:
        """Vertical bar chart of recent trade P&L."""
        n = len(entries)
        if n == 0:
            return ""
        w = max(400, n * 22)
        h = 140
        margin_top = 20
        margin_bottom = 30
        chart_h = h - margin_top - margin_bottom

        pnls = [e.pnl for e in entries]
        max_abs = max(abs(p) for p in pnls) or 1.0
        zero_y = margin_top + chart_h / 2

        bar_w = max((w - 20) / n - 2, 4)
        bars: List[str] = []
        for i, pnl in enumerate(pnls):
            x = 10 + i * (bar_w + 2)
            bar_h_px = abs(pnl) / max_abs * (chart_h / 2)
            if pnl >= 0:
                y = zero_y - bar_h_px
                color = "#4ade80"
            else:
                y = zero_y
                color = "#f87171"
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(bar_h_px, 1):.1f}" fill="{color}" rx="1"/>'
            )

        # zero line
        bars.append(
            f'<line x1="0" y1="{zero_y}" x2="{w}" y2="{zero_y}" '
            f'stroke="#475569" stroke-width="1"/>'
        )

        return f"""<div class="chart-box">
<h3>Recent Trade P&amp;L</h3>
<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">
{"".join(bars)}
</svg>
</div>"""

    @staticmethod
    def _svg_signal_bars(rows: List[SignalQualityRow]) -> str:
        """Grouped bar chart comparing IC and SNR across experiments."""
        n = len(rows)
        if n == 0:
            return ""
        w = max(300, n * 70)
        h = 140
        margin_top = 20
        margin_bottom = 35
        chart_h = h - margin_top - margin_bottom

        ics = [r.ic for r in rows]
        snrs = [r.snr for r in rows]
        max_val = max(max(abs(v) for v in ics) if ics else 0.01,
                      max(abs(v) for v in snrs) if snrs else 0.01) or 0.01

        group_w = (w - 20) / n
        bar_w = group_w * 0.35
        bars: List[str] = []

        for i, row in enumerate(rows):
            gx = 10 + i * group_w

            # IC bar
            ic_h = abs(row.ic) / max_val * chart_h
            bars.append(
                f'<rect x="{gx:.1f}" y="{margin_top + chart_h - ic_h:.1f}" '
                f'width="{bar_w:.1f}" height="{max(ic_h, 1):.1f}" '
                f'fill="#38bdf8" rx="2"/>'
            )

            # SNR bar
            snr_h = abs(row.snr) / max_val * chart_h
            bars.append(
                f'<rect x="{gx + bar_w + 2:.1f}" y="{margin_top + chart_h - snr_h:.1f}" '
                f'width="{bar_w:.1f}" height="{max(snr_h, 1):.1f}" '
                f'fill="#818cf8" rx="2"/>'
            )

            # Label
            bars.append(
                f'<text x="{gx + group_w / 2:.1f}" y="{h - 5}" '
                f'text-anchor="middle" fill="#94a3b8" font-size="9">'
                f'{row.experiment_id}</text>'
            )

        # Legend
        bars.append(
            f'<rect x="{w - 120}" y="2" width="8" height="8" fill="#38bdf8" rx="1"/>'
            f'<text x="{w - 108}" y="10" fill="#94a3b8" font-size="9">IC</text>'
            f'<rect x="{w - 80}" y="2" width="8" height="8" fill="#818cf8" rx="1"/>'
            f'<text x="{w - 68}" y="10" fill="#94a3b8" font-size="9">SNR</text>'
        )

        return f"""<div class="chart-box">
<h3>Signal Quality</h3>
<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">
{"".join(bars)}
</svg>
</div>"""
