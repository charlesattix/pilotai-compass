"""Position reconciler — compares internal paper-trade position tracker
against broker-reported state and detects/corrects discrepancies.

Discrepancy types:
  - missing_fill: internal shows a position that broker does not
  - phantom_position: broker shows a position that internal does not
  - quantity_mismatch: both sides have the position but quantities differ
  - price_drift: average entry price differs beyond tolerance

Auto-correction:
  - Minor drifts (≤ tolerance): silently correct internal state
  - Major drifts: flag for manual review, never auto-correct

Dry-run mode: report what WOULD be corrected without mutating anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────
DEFAULT_QTY_TOLERANCE = 2          # auto-correct if mismatch ≤ 2 contracts
DEFAULT_PRICE_TOLERANCE_PCT = 1.0  # auto-correct if avg price within 1%
DEFAULT_NOTIONAL_TOLERANCE = 500   # auto-correct if notional diff ≤ $500


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class PositionRecord:
    """Unified position representation from either source."""
    symbol: str
    quantity: int              # signed: positive=long, negative=short
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealised_pnl: float = 0.0
    side: str = ""             # "long" or "short"


@dataclass
class Discrepancy:
    """A single detected discrepancy between internal and broker."""
    symbol: str
    disc_type: str             # missing_fill, phantom_position, quantity_mismatch, price_drift
    severity: str              # minor, major
    internal_qty: int
    broker_qty: int
    internal_price: float
    broker_price: float
    detail: str
    auto_corrected: bool = False
    correction_applied: str = ""


@dataclass
class ReconciliationReport:
    """Full reconciliation output."""
    timestamp: str
    n_internal: int
    n_broker: int
    n_matched: int
    n_discrepancies: int
    n_auto_corrected: int
    n_flagged: int
    discrepancies: List[Discrepancy] = field(default_factory=list)
    matched_symbols: List[str] = field(default_factory=list)
    is_clean: bool = True      # True if zero discrepancies
    dry_run: bool = False


@dataclass
class ReconcilerConfig:
    """Reconciler configuration."""
    qty_tolerance: int = DEFAULT_QTY_TOLERANCE
    price_tolerance_pct: float = DEFAULT_PRICE_TOLERANCE_PCT
    notional_tolerance: float = DEFAULT_NOTIONAL_TOLERANCE
    auto_correct: bool = True
    dry_run: bool = False


# ── Core reconciler ────────────────────────────────────────────────────────
class PositionReconciler:
    """Reconciles internal position tracker against broker state."""

    def __init__(self, config: Optional[ReconcilerConfig] = None) -> None:
        self.config = config or ReconcilerConfig()
        self._history: List[ReconciliationReport] = []

    def reconcile(
        self,
        internal: List[PositionRecord],
        broker: List[PositionRecord],
        dry_run: Optional[bool] = None,
    ) -> ReconciliationReport:
        """Compare internal vs broker positions and detect discrepancies.

        Parameters
        ----------
        internal : list of PositionRecord from paper trading engine
        broker : list of PositionRecord from Alpaca/IBKR API
        dry_run : override config dry_run for this call

        Returns
        -------
        ReconciliationReport with all discrepancies and corrections.
        """
        is_dry = dry_run if dry_run is not None else self.config.dry_run
        now = _now()

        int_map = {p.symbol: p for p in internal}
        brk_map = {p.symbol: p for p in broker}

        all_symbols = sorted(set(int_map.keys()) | set(brk_map.keys()))
        discrepancies: List[Discrepancy] = []
        matched: List[str] = []

        for sym in all_symbols:
            int_pos = int_map.get(sym)
            brk_pos = brk_map.get(sym)

            if int_pos and not brk_pos:
                # Missing fill: internal has it, broker doesn't
                disc = Discrepancy(
                    symbol=sym, disc_type="missing_fill",
                    severity=self._classify_severity(abs(int_pos.quantity), 0),
                    internal_qty=int_pos.quantity, broker_qty=0,
                    internal_price=int_pos.avg_price, broker_price=0.0,
                    detail=f"Internal has {int_pos.quantity} {sym}, broker has none",
                )
                if self._is_minor(disc) and self.config.auto_correct and not is_dry:
                    disc.auto_corrected = True
                    disc.correction_applied = f"Removed {sym} from internal tracker"
                discrepancies.append(disc)

            elif brk_pos and not int_pos:
                # Phantom position: broker has it, internal doesn't
                disc = Discrepancy(
                    symbol=sym, disc_type="phantom_position",
                    severity=self._classify_severity(0, abs(brk_pos.quantity)),
                    internal_qty=0, broker_qty=brk_pos.quantity,
                    internal_price=0.0, broker_price=brk_pos.avg_price,
                    detail=f"Broker has {brk_pos.quantity} {sym}, internal has none",
                )
                if self._is_minor(disc) and self.config.auto_correct and not is_dry:
                    disc.auto_corrected = True
                    disc.correction_applied = f"Added {sym} qty={brk_pos.quantity} to internal"
                discrepancies.append(disc)

            else:
                # Both sides have it — check for mismatches
                assert int_pos is not None and brk_pos is not None
                qty_diff = abs(int_pos.quantity - brk_pos.quantity)
                price_diff_pct = (
                    abs(int_pos.avg_price - brk_pos.avg_price) / max(brk_pos.avg_price, 0.01) * 100
                )

                if qty_diff > 0:
                    disc = Discrepancy(
                        symbol=sym, disc_type="quantity_mismatch",
                        severity=self._classify_severity(qty_diff, 0),
                        internal_qty=int_pos.quantity, broker_qty=brk_pos.quantity,
                        internal_price=int_pos.avg_price, broker_price=brk_pos.avg_price,
                        detail=f"{sym}: internal={int_pos.quantity}, broker={brk_pos.quantity} (diff={qty_diff})",
                    )
                    if self._is_minor(disc) and self.config.auto_correct and not is_dry:
                        disc.auto_corrected = True
                        disc.correction_applied = f"Adjusted {sym} qty {int_pos.quantity}→{brk_pos.quantity}"
                    discrepancies.append(disc)

                elif price_diff_pct > self.config.price_tolerance_pct:
                    disc = Discrepancy(
                        symbol=sym, disc_type="price_drift",
                        severity="minor" if price_diff_pct <= self.config.price_tolerance_pct * 3 else "major",
                        internal_qty=int_pos.quantity, broker_qty=brk_pos.quantity,
                        internal_price=int_pos.avg_price, broker_price=brk_pos.avg_price,
                        detail=f"{sym}: price drift {price_diff_pct:.1f}% (int={int_pos.avg_price:.2f}, brk={brk_pos.avg_price:.2f})",
                    )
                    if disc.severity == "minor" and self.config.auto_correct and not is_dry:
                        disc.auto_corrected = True
                        disc.correction_applied = f"Updated {sym} price {int_pos.avg_price:.2f}→{brk_pos.avg_price:.2f}"
                    discrepancies.append(disc)

                else:
                    matched.append(sym)

        n_auto = sum(1 for d in discrepancies if d.auto_corrected)
        n_flagged = sum(1 for d in discrepancies if not d.auto_corrected)

        report = ReconciliationReport(
            timestamp=now,
            n_internal=len(internal),
            n_broker=len(broker),
            n_matched=len(matched),
            n_discrepancies=len(discrepancies),
            n_auto_corrected=n_auto,
            n_flagged=n_flagged,
            discrepancies=discrepancies,
            matched_symbols=matched,
            is_clean=len(discrepancies) == 0,
            dry_run=is_dry,
        )
        self._history.append(report)
        return report

    def apply_corrections(
        self,
        internal: List[PositionRecord],
        report: ReconciliationReport,
        broker: List[PositionRecord],
    ) -> List[PositionRecord]:
        """Apply auto-corrections from a reconciliation report to internal positions.

        Only applies corrections flagged as auto_corrected=True.
        Returns a new list (does not mutate input).
        """
        if report.dry_run:
            return list(internal)

        result_map = {p.symbol: PositionRecord(
            symbol=p.symbol, quantity=p.quantity, avg_price=p.avg_price,
            market_value=p.market_value, unrealised_pnl=p.unrealised_pnl, side=p.side,
        ) for p in internal}

        brk_map = {p.symbol: p for p in broker}

        for disc in report.discrepancies:
            if not disc.auto_corrected:
                continue

            if disc.disc_type == "missing_fill":
                result_map.pop(disc.symbol, None)

            elif disc.disc_type == "phantom_position":
                bp = brk_map.get(disc.symbol)
                if bp:
                    result_map[disc.symbol] = PositionRecord(
                        symbol=bp.symbol, quantity=bp.quantity,
                        avg_price=bp.avg_price, market_value=bp.market_value,
                        unrealised_pnl=bp.unrealised_pnl, side=bp.side,
                    )

            elif disc.disc_type == "quantity_mismatch":
                if disc.symbol in result_map:
                    result_map[disc.symbol].quantity = disc.broker_qty

            elif disc.disc_type == "price_drift":
                if disc.symbol in result_map:
                    result_map[disc.symbol].avg_price = disc.broker_price

        return list(result_map.values())

    @property
    def history(self) -> List[ReconciliationReport]:
        return list(self._history)

    def generate_report(
        self,
        report: ReconciliationReport,
        output_path: str = "reports/reconciliation.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(report)
        path.write_text(html, encoding="utf-8")
        return path

    # ── Internals ───────────────────────────────────────────────────────────
    def _classify_severity(self, qty_diff: int, broker_qty: int) -> str:
        """Minor if within tolerance, major otherwise."""
        relevant = max(qty_diff, broker_qty)
        if relevant <= self.config.qty_tolerance:
            return "minor"
        return "major"

    def _is_minor(self, disc: Discrepancy) -> bool:
        return disc.severity == "minor"

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: ReconciliationReport) -> str:
        status_cls = "pos" if r.is_clean else "neg"
        status_txt = "CLEAN" if r.is_clean else f"{r.n_discrepancies} DISCREPANCIES"
        mode = " (DRY RUN)" if r.dry_run else ""

        disc_rows = ""
        for d in r.discrepancies:
            sev_cls = "warn" if d.severity == "minor" else "neg"
            corr = f'<span class="pos">{d.correction_applied}</span>' if d.auto_corrected else '<span class="neg">FLAGGED</span>'
            disc_rows += (
                f"<tr><td>{d.symbol}</td><td>{d.disc_type}</td>"
                f'<td class="{sev_cls}">{d.severity}</td>'
                f"<td>{d.internal_qty}</td><td>{d.broker_qty}</td>"
                f"<td>{d.internal_price:.2f}</td><td>{d.broker_price:.2f}</td>"
                f"<td>{corr}</td></tr>"
            )

        matched_list = ", ".join(r.matched_symbols) if r.matched_symbols else "—"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Position Reconciliation{mode}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#fff;color:#1e293b;padding:24px;max-width:900px;margin:0 auto}}
h1{{font-size:1.4rem;margin-bottom:4px}}
h2{{font-size:1rem;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin:18px 0 8px}}
.sub{{color:#64748b;font-size:.85rem;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:18px}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px}}
.card .lbl{{font-size:.7rem;color:#64748b;text-transform:uppercase}}
.card .val{{font-size:1.2rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:14px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #e2e8f0}}
th{{color:#64748b;background:#f8fafc}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}.warn{{color:#d97706}}
</style>
</head>
<body>
<h1>Position Reconciliation{mode}</h1>
<p class="sub">{r.timestamp}</p>

<div class="grid">
<div class="card"><div class="lbl">Status</div><div class="val {status_cls}">{status_txt}</div></div>
<div class="card"><div class="lbl">Internal</div><div class="val">{r.n_internal}</div></div>
<div class="card"><div class="lbl">Broker</div><div class="val">{r.n_broker}</div></div>
<div class="card"><div class="lbl">Matched</div><div class="val pos">{r.n_matched}</div></div>
<div class="card"><div class="lbl">Discrepancies</div><div class="val {'neg' if r.n_discrepancies else ''}">{r.n_discrepancies}</div></div>
<div class="card"><div class="lbl">Auto-Fixed</div><div class="val">{r.n_auto_corrected}</div></div>
<div class="card"><div class="lbl">Flagged</div><div class="val {'neg' if r.n_flagged else ''}">{r.n_flagged}</div></div>
</div>

<h2>Matched Positions</h2>
<p>{matched_list}</p>

<h2>Discrepancies</h2>
{'<table><thead><tr><th>Symbol</th><th>Type</th><th>Severity</th><th>Int Qty</th><th>Brk Qty</th><th>Int Price</th><th>Brk Price</th><th>Action</th></tr></thead><tbody>' + disc_rows + '</tbody></table>' if disc_rows else '<p class="pos">No discrepancies found.</p>'}

</body>
</html>"""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
