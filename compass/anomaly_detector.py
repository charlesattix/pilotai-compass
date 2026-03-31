"""Anomaly detection system – detects anomalous market conditions and trade
outcomes using z-score and IQR methods with severity classification and
Telegram-ready alert messages.

Provides:
  1. Market anomalies: unusual volume, price gaps, IV spikes
  2. Trade anomalies: outlier P&L, abnormal slippage, unusual fills
  3. Z-score and IQR-based detection
  4. Severity classification (info / warning / critical)
  5. Telegram-ready alert message formatting
  6. Anomaly history tracking
  7. HTML dashboard with timeline, distribution, and alerts table
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Severity ────────────────────────────────────────────────────────────────
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"

SEVERITIES = [INFO, WARNING, CRITICAL]

# ── Default thresholds ──────────────────────────────────────────────────────
DEFAULT_ZSCORE_WARN = 2.0
DEFAULT_ZSCORE_CRIT = 3.0
DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_LOOKBACK = 60


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class AnomalyConfig:
    """Detection thresholds."""
    zscore_warn: float = DEFAULT_ZSCORE_WARN
    zscore_crit: float = DEFAULT_ZSCORE_CRIT
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
    lookback: int = DEFAULT_LOOKBACK


@dataclass
class Anomaly:
    """A single detected anomaly."""
    timestamp: str
    category: str          # "market" or "trade"
    anomaly_type: str      # e.g. "price_gap", "iv_spike", "outlier_pnl"
    severity: str          # info / warning / critical
    value: float           # observed value
    threshold: float       # threshold that was breached
    zscore: float = 0.0
    detail: str = ""


@dataclass
class AlertMessage:
    """Telegram-ready alert."""
    severity: str
    title: str
    body: str
    timestamp: str


@dataclass
class AnomalyStats:
    """Aggregate anomaly statistics."""
    total: int = 0
    info_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Complete anomaly detection output."""
    anomalies: List[Anomaly] = field(default_factory=list)
    alerts: List[AlertMessage] = field(default_factory=list)
    stats: Optional[AnomalyStats] = None
    generated_at: str = ""


# ── Core detector ───────────────────────────────────────────────────────────
class AnomalyDetector:
    """Detects anomalous market conditions and trade outcomes."""

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.config = config or AnomalyConfig()
        self._history: List[Anomaly] = []

    # ── Public API ──────────────────────────────────────────────────────────
    def detect(
        self,
        market_data: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
    ) -> AnomalyResult:
        """Run anomaly detection on market data and/or trades.

        Parameters
        ----------
        market_data : pd.DataFrame, optional
            Columns: date, price/close, volume.  Optional: iv, vix.
        trades : pd.DataFrame, optional
            Columns: date, pnl.  Optional: slippage, fill_time.
        """
        anomalies: List[Anomaly] = []

        if market_data is not None and not market_data.empty:
            anomalies.extend(self._detect_market(market_data))

        if trades is not None and not trades.empty:
            anomalies.extend(self._detect_trades(trades))

        self._history.extend(anomalies)

        alerts = [self._to_alert(a) for a in anomalies if a.severity in (WARNING, CRITICAL)]
        stats = self._compute_stats(anomalies)

        return AnomalyResult(
            anomalies=anomalies,
            alerts=alerts,
            stats=stats,
            generated_at=self._now(),
        )

    def get_history(self) -> List[Anomaly]:
        """Return full anomaly history across calls."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def generate_report(
        self,
        result: AnomalyResult,
        output_path: str | Path = "reports/anomaly_detection.html",
    ) -> Path:
        """Write self-contained HTML dashboard."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Anomaly report written to %s", path)
        return path

    # ── Market anomalies ────────────────────────────────────────────────────
    def _detect_market(self, df: pd.DataFrame) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        price_col = "close" if "close" in df.columns else "price" if "price" in df.columns else None

        # Price gaps
        if price_col:
            anomalies.extend(self._check_series(
                df, price_col, "market", "price_gap",
                use_returns=True,
            ))

        # Volume anomalies
        if "volume" in df.columns:
            anomalies.extend(self._check_series(
                df, "volume", "market", "unusual_volume",
            ))

        # IV spikes
        iv_col = "iv" if "iv" in df.columns else "vix" if "vix" in df.columns else None
        if iv_col:
            anomalies.extend(self._check_series(
                df, iv_col, "market", "iv_spike",
            ))

        return anomalies

    # ── Trade anomalies ─────────────────────────────────────────────────────
    def _detect_trades(self, df: pd.DataFrame) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if "pnl" in df.columns:
            anomalies.extend(self._check_series(
                df, "pnl", "trade", "outlier_pnl",
            ))

        if "slippage" in df.columns:
            anomalies.extend(self._check_series(
                df, "slippage", "trade", "abnormal_slippage",
            ))

        if "fill_time" in df.columns:
            anomalies.extend(self._check_series(
                df, "fill_time", "trade", "unusual_fill_time",
            ))

        return anomalies

    # ── Detection engine ────────────────────────────────────────────────────
    def _check_series(
        self,
        df: pd.DataFrame,
        col: str,
        category: str,
        anomaly_type: str,
        use_returns: bool = False,
    ) -> List[Anomaly]:
        vals = df[col].dropna()
        if len(vals) < 10:
            return []

        if use_returns:
            vals = vals.pct_change().dropna()

        anomalies: List[Anomaly] = []
        lb = self.config.lookback

        for i in range(lb, len(vals)):
            window = vals.iloc[max(0, i - lb):i]
            current = float(vals.iloc[i])
            idx = vals.index[i]
            ts = str(idx)

            # Z-score method
            mean = float(window.mean())
            std = float(window.std())
            if std > 1e-12:
                z = (current - mean) / std
            else:
                z = 0.0

            severity = self._classify_zscore(abs(z))
            if severity is None:
                # IQR fallback
                severity = self._classify_iqr(current, window)

            if severity is not None:
                threshold = mean + self.config.zscore_warn * std if std > 1e-12 else mean
                anomalies.append(Anomaly(
                    timestamp=ts,
                    category=category,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    value=current,
                    threshold=threshold,
                    zscore=z,
                    detail=f"{col}={current:.4f}, z={z:.2f}",
                ))

        return anomalies

    def _classify_zscore(self, abs_z: float) -> Optional[str]:
        if abs_z >= self.config.zscore_crit:
            return CRITICAL
        if abs_z >= self.config.zscore_warn:
            return WARNING
        return None

    def _classify_iqr(self, value: float, window: pd.Series) -> Optional[str]:
        q1 = float(window.quantile(0.25))
        q3 = float(window.quantile(0.75))
        iqr = q3 - q1
        if iqr < 1e-12:
            return None
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        if value < lower or value > upper:
            extreme_lower = q1 - 3.0 * iqr
            extreme_upper = q3 + 3.0 * iqr
            if value < extreme_lower or value > extreme_upper:
                return CRITICAL
            return WARNING
        return None

    # ── Alert formatting ────────────────────────────────────────────────────
    @staticmethod
    def _to_alert(a: Anomaly) -> AlertMessage:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(a.severity, "")
        title = f"{icon} {a.severity.upper()}: {a.anomaly_type.replace('_', ' ').title()}"
        body = (
            f"Category: {a.category}\n"
            f"Value: {a.value:.4f} (z-score: {a.zscore:.2f})\n"
            f"Threshold: {a.threshold:.4f}\n"
            f"Time: {a.timestamp}"
        )
        return AlertMessage(
            severity=a.severity,
            title=title,
            body=body,
            timestamp=a.timestamp,
        )

    @staticmethod
    def format_telegram(alert: AlertMessage) -> str:
        """Format alert as Telegram message text."""
        return f"*{alert.title}*\n```\n{alert.body}\n```"

    # ── Stats ───────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_stats(anomalies: List[Anomaly]) -> AnomalyStats:
        by_type: Dict[str, int] = {}
        by_cat: Dict[str, int] = {}
        info_c = warn_c = crit_c = 0
        for a in anomalies:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_cat[a.category] = by_cat.get(a.category, 0) + 1
            if a.severity == INFO:
                info_c += 1
            elif a.severity == WARNING:
                warn_c += 1
            elif a.severity == CRITICAL:
                crit_c += 1
        return AnomalyStats(
            total=len(anomalies),
            info_count=info_c,
            warning_count=warn_c,
            critical_count=crit_c,
            by_type=by_type,
            by_category=by_cat,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML dashboard ──────────────────────────────────────────────────────
    def _build_html(self, r: AnomalyResult) -> str:
        cards = self._html_cards(r)
        timeline = self._html_timeline(r.anomalies)
        severity_dist = self._html_severity(r.stats)
        type_tbl = self._html_type_breakdown(r.stats)
        alerts_tbl = self._html_alerts(r.alerts)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Anomaly Detection</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.info{{color:#38bdf8}}.warning{{color:#fbbf24}}.critical{{color:#f87171}}
</style>
</head>
<body>
<h1>Anomaly Detection Dashboard</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{cards}
{severity_dist}
{type_tbl}
{alerts_tbl}
{timeline}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: AnomalyResult) -> str:
        s = r.stats or AnomalyStats()
        return f"""<div class="grid">
<div class="card"><div class="lbl">Total Anomalies</div><div class="val">{s.total}</div></div>
<div class="card"><div class="lbl">Critical</div><div class="val critical">{s.critical_count}</div></div>
<div class="card"><div class="lbl">Warning</div><div class="val warning">{s.warning_count}</div></div>
<div class="card"><div class="lbl">Info</div><div class="val info">{s.info_count}</div></div>
<div class="card"><div class="lbl">Alerts Sent</div><div class="val">{len(r.alerts)}</div></div>
</div>"""

    @staticmethod
    def _html_severity(stats: Optional[AnomalyStats]) -> str:
        if not stats or stats.total == 0:
            return ""
        rows = ""
        for sev, cnt in [("critical", stats.critical_count), ("warning", stats.warning_count), ("info", stats.info_count)]:
            pct = cnt / stats.total * 100 if stats.total > 0 else 0
            rows += f'<tr><td class="{sev}">{sev.upper()}</td><td>{cnt}</td><td>{pct:.0f}%</td></tr>'
        return f"""<div class="sec">
<h2>Severity Distribution</h2>
<table><thead><tr><th>Severity</th><th>Count</th><th>%</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""

    @staticmethod
    def _html_type_breakdown(stats: Optional[AnomalyStats]) -> str:
        if not stats or not stats.by_type:
            return ""
        rows = "".join(
            f"<tr><td>{t}</td><td>{c}</td></tr>"
            for t, c in sorted(stats.by_type.items(), key=lambda x: -x[1])
        )
        return f"""<div class="sec">
<h2>Anomaly Type Breakdown</h2>
<table><thead><tr><th>Type</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""

    @staticmethod
    def _html_alerts(alerts: List[AlertMessage]) -> str:
        if not alerts:
            return ""
        rows = ""
        for a in alerts[:30]:
            rows += (
                f'<tr><td class="{a.severity}">{a.severity.upper()}</td>'
                f"<td>{a.title}</td>"
                f"<td><pre style='margin:0;font-size:.8rem;white-space:pre-wrap'>{a.body}</pre></td>"
                f"<td>{a.timestamp}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Recent Alerts</h2>
<table><thead><tr><th>Severity</th><th>Title</th><th>Detail</th><th>Time</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""

    @staticmethod
    def _html_timeline(anomalies: List[Anomaly]) -> str:
        if not anomalies:
            return ""
        rows = ""
        for a in anomalies[:50]:
            rows += (
                f'<tr><td>{a.timestamp}</td>'
                f'<td class="{a.severity}">{a.severity.upper()}</td>'
                f"<td>{a.category}</td>"
                f"<td>{a.anomaly_type}</td>"
                f"<td>{a.value:.4f}</td>"
                f"<td>{a.zscore:.2f}</td>"
                f"<td>{a.detail}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Anomaly Timeline</h2>
<table><thead><tr><th>Time</th><th>Severity</th><th>Category</th><th>Type</th><th>Value</th><th>Z-Score</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""
