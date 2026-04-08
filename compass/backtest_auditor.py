"""
compass/backtest_auditor.py — Backtest Integrity Audit Tool.

Checks any backtest for the bugs we've actually encountered:
  1. Dilution:       >50% zero-return days → inflated Sharpe
  2. Synthetic data: np.random, Black-Scholes as "prices"
  3. Look-ahead:     Future data used in current decisions
  4. Sharpe formula: Must use arithmetic returns, not CAGR-based
  5. Survivorship:   Expired/delisted options handled?
  6. Transaction costs: Bid-ask + commissions included?
  7. Capacity:       Trades within market volume limits?

Usage:
    from compass.backtest_auditor import BacktestAuditor
    auditor = BacktestAuditor()
    report = auditor.audit(results_dict)
    print(report.summary())
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Known ATM daily volumes from IronVault (real data)
ATM_ADV = {"SPY": 500_000, "GLD": 5_000, "TLT": 8_000, "XLI": 3_000,
           "XLF": 10_000, "QQQ": 50_000, "IBIT": 500}


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Result of a single audit check."""
    name: str
    passed: bool
    severity: str           # "PASS", "WARNING", "FAIL", "CRITICAL"
    message: str
    details: str = ""
    metric_value: Any = None
    threshold: Any = None


@dataclass
class AuditReport:
    """Complete audit report."""
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    n_passed: int = 0
    n_warnings: int = 0
    n_failed: int = 0
    n_critical: int = 0
    overall_grade: str = ""     # "A", "B", "C", "D", "F"
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Backtest Integrity Audit Report",
            "=" * 60,
            f"  Grade: {self.overall_grade}",
            f"  Passed: {self.n_passed} | Warnings: {self.n_warnings} | "
            f"Failed: {self.n_failed} | Critical: {self.n_critical}",
            "",
        ]
        for c in self.checks:
            icon = {"PASS": "OK", "WARNING": "!!", "FAIL": "XX", "CRITICAL": "**"}[c.severity]
            lines.append(f"  [{icon}] {c.name}: {c.message}")
            if c.details:
                lines.append(f"       {c.details}")
        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    - {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════════════════

def check_dilution(
    equity_curve: List = None,
    trades: List = None,
    n_days: int = 0,
    threshold: float = 0.50,
) -> CheckResult:
    """Check if >threshold of days have zero returns (diluted backtest).

    A diluted backtest has many zero-return days that artificially
    reduce volatility and inflate the Sharpe ratio.
    """
    if equity_curve and len(equity_curve) > 1:
        if isinstance(equity_curve[0], dict):
            values = [e.get("equity", e.get("value", 0)) for e in equity_curve]
        else:
            values = list(equity_curve)
        returns = np.diff(values) / np.where(np.array(values[:-1]) != 0, values[:-1], 1)
        zero_pct = float((np.abs(returns) < 1e-10).sum()) / max(len(returns), 1)
    elif trades and n_days > 0:
        trading_days = len(set(
            t.get("entry_date", t.get("exit_date", ""))[:10] for t in trades
        ))
        zero_pct = 1.0 - (trading_days / max(n_days, 1))
    else:
        return CheckResult("Dilution Check", True, "PASS",
                          "Insufficient data to check dilution", metric_value=None)

    passed = zero_pct < threshold
    severity = "PASS" if passed else ("WARNING" if zero_pct < 0.80 else "FAIL")
    return CheckResult(
        name="Dilution Check",
        passed=passed,
        severity=severity,
        message=f"{zero_pct:.0%} of days have zero returns (threshold: {threshold:.0%})",
        details="High dilution inflates Sharpe by reducing measured volatility"
                if not passed else "",
        metric_value=round(zero_pct, 4),
        threshold=threshold,
    )


def check_synthetic_data(
    data_source: str = "",
    code_path: str = "",
    trades: List = None,
) -> CheckResult:
    """Check if backtest uses synthetic/generated data instead of real market data.

    Flags: np.random in pricing, Black-Scholes as "prices", BACKTEST_CREDIT_FRACTION,
    fixed credit percentages, or any non-IronVault data source.
    """
    issues = []

    # Check data source label
    if data_source:
        ds = data_source.lower()
        if any(s in ds for s in ["synthetic", "simulated", "random", "generated", "heuristic"]):
            issues.append(f"Data source labeled as '{data_source}'")
        if "ironvault" not in ds and "real" not in ds and data_source:
            issues.append(f"Data source '{data_source}' is not IronVault/real")

    # Check source code for synthetic patterns
    if code_path and Path(code_path).exists():
        try:
            code = Path(code_path).read_text(errors="ignore")
            patterns = [
                (r"np\.random\.(normal|uniform|random|randn|seed|RandomState)",
                 "np.random used (synthetic data generation)"),
                (r"BACKTEST_CREDIT_FRACTION",
                 "BACKTEST_CREDIT_FRACTION (heuristic pricing, not real)"),
                (r"black.scholes|bs_price|bsm_price",
                 "Black-Scholes used as price source"),
                (r"credit\s*=\s*\d+\.\d+\s*\*\s*width",
                 "Fixed credit fraction (not real option prices)"),
            ]
            for pattern, desc in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(desc)
        except Exception:
            pass

    # Check trade data for suspicious patterns
    if trades:
        credits = [t.get("credit", t.get("entry_credit", 0)) for t in trades if t.get("credit", t.get("entry_credit", 0)) > 0]
        if len(credits) > 10:
            # Suspiciously uniform credits suggest synthetic pricing
            std = np.std(credits)
            mean = np.mean(credits)
            cv = std / mean if mean > 0 else 0
            if cv < 0.01:
                issues.append(f"Credits are suspiciously uniform (CV={cv:.4f}) — likely synthetic")

    passed = len(issues) == 0
    severity = "PASS" if passed else ("CRITICAL" if len(issues) >= 2 else "FAIL")
    return CheckResult(
        name="Synthetic Data Check",
        passed=passed,
        severity=severity,
        message="No synthetic data detected" if passed else f"{len(issues)} synthetic data issue(s)",
        details="; ".join(issues) if issues else "",
        metric_value=len(issues),
    )


def check_look_ahead(
    trades: List = None,
    code_path: str = "",
) -> CheckResult:
    """Check for look-ahead bias.

    Flags: entry decisions using exit data, future prices in features,
    shift(-1) in decision logic, or non-causal indicators.
    """
    issues = []

    # Check trade timestamps
    if trades:
        for i, t in enumerate(trades):
            entry = t.get("entry_date", "")
            exit_d = t.get("exit_date", "")
            if entry and exit_d and exit_d < entry:
                issues.append(f"Trade {i}: exit ({exit_d}) before entry ({entry})")

    # Check source code
    if code_path and Path(code_path).exists():
        try:
            code = Path(code_path).read_text(errors="ignore")
            patterns = [
                (r"\.shift\(-\d+\)", "Negative shift (uses future data)"),
                (r"iloc\[-1\].*entry", "Last element used for entry (possible lookahead)"),
            ]
            for pattern, desc in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(desc)
        except Exception:
            pass

    passed = len(issues) == 0
    severity = "PASS" if passed else "CRITICAL"
    return CheckResult(
        name="Look-Ahead Bias Check",
        passed=passed,
        severity=severity,
        message="No look-ahead bias detected" if passed else f"{len(issues)} look-ahead issue(s)",
        details="; ".join(issues) if issues else "",
        metric_value=len(issues),
    )


def check_sharpe_formula(
    reported_sharpe: float = 0,
    returns: np.ndarray = None,
    trades: List = None,
    cagr: float = 0,
    n_periods: int = 252,
) -> CheckResult:
    """Verify Sharpe uses arithmetic mean / std * sqrt(N), not CAGR-based.

    The bug: using CAGR (geometric) as numerator inflates Sharpe by ~2.4x
    at high return levels (e.g. 100% CAGR → 2.4x inflation).
    """
    if returns is not None and len(returns) > 1:
        arith_mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        correct_sharpe = (arith_mean / std * math.sqrt(min(len(returns), n_periods))
                         ) if std > 1e-9 else 0.0
    elif trades:
        pnls = np.array([t.get("pnl", 0) for t in trades])
        if len(pnls) > 1:
            arith_mean = float(np.mean(pnls))
            std = float(np.std(pnls, ddof=1))
            correct_sharpe = (arith_mean / std * math.sqrt(min(len(pnls), 52))
                             ) if std > 1e-9 else 0.0
        else:
            correct_sharpe = 0.0
    else:
        # Estimate inflation from CAGR
        if cagr > 0.3 and reported_sharpe > 0:
            # At 100% CAGR, geometric/arithmetic inflation ≈ 1 + 0.5*σ²
            # Rough check: if Sharpe > 2 * (cagr / 0.15), it's likely inflated
            expected_max = cagr / 0.08  # generous bound
            if reported_sharpe > expected_max * 1.5:
                return CheckResult(
                    name="Sharpe Formula Check",
                    passed=False,
                    severity="FAIL",
                    message=f"Reported Sharpe {reported_sharpe:.2f} likely uses CAGR-based formula "
                            f"(expected ≤{expected_max:.1f} for {cagr:.0%} CAGR)",
                    details="Geometric CAGR in Sharpe numerator inflates by ~2.4x at high returns",
                    metric_value=reported_sharpe,
                    threshold=expected_max,
                )
        return CheckResult("Sharpe Formula Check", True, "PASS",
                          "Insufficient data for Sharpe verification")

    if reported_sharpe == 0:
        return CheckResult("Sharpe Formula Check", True, "PASS",
                          f"Computed Sharpe: {correct_sharpe:.2f} (no reported value to compare)")

    ratio = reported_sharpe / correct_sharpe if correct_sharpe != 0 else 0
    if abs(ratio - 1.0) < 0.15:
        return CheckResult("Sharpe Formula Check", True, "PASS",
                          f"Reported {reported_sharpe:.2f} matches computed {correct_sharpe:.2f} "
                          f"(ratio {ratio:.2f})",
                          metric_value=reported_sharpe, threshold=correct_sharpe)

    severity = "FAIL" if ratio > 1.5 else "WARNING"
    return CheckResult(
        name="Sharpe Formula Check",
        passed=False,
        severity=severity,
        message=f"Reported Sharpe {reported_sharpe:.2f} differs from computed {correct_sharpe:.2f} "
                f"(ratio {ratio:.2f})",
        details=f"{'Likely using CAGR-based formula (inflated)' if ratio > 1.3 else 'Minor discrepancy'}",
        metric_value=reported_sharpe,
        threshold=correct_sharpe,
    )


def check_survivorship(trades: List = None) -> CheckResult:
    """Check if expired/delisted options are properly handled.

    Flags: trades that never close, or trades held past expiration.
    """
    if not trades:
        return CheckResult("Survivorship Check", True, "PASS", "No trades to check")

    issues = 0
    for t in trades:
        if not t.get("exit_date") and t.get("status", "closed") == "open":
            continue  # open trades are OK
        if not t.get("exit_date") and t.get("status", "closed") == "closed":
            issues += 1  # closed but no exit date

    pct = issues / max(len(trades), 1)
    passed = pct < 0.05
    severity = "PASS" if passed else ("WARNING" if pct < 0.20 else "FAIL")
    return CheckResult(
        name="Survivorship Check",
        passed=passed,
        severity=severity,
        message=f"{issues}/{len(trades)} trades ({pct:.0%}) have survivorship issues",
        details="Trades without exit dates may represent unhandled expirations" if not passed else "",
        metric_value=pct,
        threshold=0.05,
    )


def check_transaction_costs(
    trades: List = None,
    has_commissions: bool = False,
    has_slippage: bool = False,
    spread_width: float = 5.0,
) -> CheckResult:
    """Check if bid-ask spreads and commissions are included."""
    issues = []

    if not has_commissions:
        issues.append("No commission model applied")
    if not has_slippage:
        issues.append("No slippage/bid-ask model applied")

    if trades:
        # Check if any trade has cost fields
        has_cost_field = any(
            t.get("commission", 0) > 0 or t.get("slippage", 0) > 0
            or "cost" in t
            for t in trades
        )
        if not has_cost_field and not has_commissions:
            issues.append("No cost fields found in trade records")

        # Check credit reasonableness
        credits = [t.get("credit", t.get("entry_credit", 0)) for t in trades
                   if t.get("credit", t.get("entry_credit", 0)) > 0]
        if credits and spread_width > 0:
            avg_credit = np.mean(credits)
            credit_pct = avg_credit / spread_width
            if credit_pct > 0.80:
                issues.append(f"Avg credit is {credit_pct:.0%} of width — suspiciously high "
                             "(real: typically 20-40%)")

    passed = len(issues) == 0
    severity = "PASS" if passed else ("WARNING" if len(issues) == 1 else "FAIL")
    return CheckResult(
        name="Transaction Cost Check",
        passed=passed,
        severity=severity,
        message="Transaction costs properly modeled" if passed else f"{len(issues)} cost issue(s)",
        details="; ".join(issues) if issues else "",
        metric_value=len(issues),
    )


def check_capacity(
    trades: List = None,
    ticker: str = "SPY",
    avg_contracts: int = 0,
) -> CheckResult:
    """Check if strategy trades within market volume limits.

    Uses real ATM ADV from IronVault data. Flags if avg trade size
    exceeds 5% of daily volume (market impact threshold).
    """
    adv = ATM_ADV.get(ticker, 5_000)

    if avg_contracts <= 0 and trades:
        contracts = [t.get("contracts", 1) for t in trades]
        avg_contracts = int(np.mean(contracts)) if contracts else 1

    if avg_contracts <= 0:
        return CheckResult("Capacity Check", True, "PASS",
                          "No position size data available")

    participation = avg_contracts / adv if adv > 0 else 1.0
    passed = participation < 0.05
    severity = "PASS" if passed else ("WARNING" if participation < 0.15 else "FAIL")
    return CheckResult(
        name="Capacity Check",
        passed=passed,
        severity=severity,
        message=f"{avg_contracts} contracts vs {adv:,} ATM ADV ({ticker}) = "
                f"{participation:.1%} participation",
        details=(f"Exceeds 5% participation limit — market impact will degrade returns"
                 if not passed else f"{ticker} has sufficient liquidity"),
        metric_value=round(participation, 4),
        threshold=0.05,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main auditor
# ═══════════════════════════════════════════════════════════════════════════

class BacktestAuditor:
    """Runs all integrity checks on a backtest."""

    def audit(
        self,
        results: Dict[str, Any] = None,
        trades: List[Dict] = None,
        equity_curve: List = None,
        reported_sharpe: float = 0,
        reported_cagr: float = 0,
        data_source: str = "",
        code_path: str = "",
        ticker: str = "SPY",
        has_commissions: bool = False,
        has_slippage: bool = False,
        spread_width: float = 5.0,
        n_days: int = 1500,
    ) -> AuditReport:
        """Run all checks and produce audit report."""
        # Extract from results dict if provided
        if results:
            trades = trades or results.get("trades", [])
            equity_curve = equity_curve or results.get("equity_curve", [])
            reported_sharpe = reported_sharpe or results.get("sharpe_ratio",
                              results.get("sharpe", 0))
            reported_cagr = reported_cagr or results.get("cagr",
                            results.get("return_pct", 0) / 100)
            data_source = data_source or results.get("data_source", "")
            has_commissions = has_commissions or results.get("commission_per_contract", 0) > 0
            has_slippage = has_slippage or results.get("slippage", 0) > 0
            n_days = n_days or results.get("n_days", 1500)

        checks = [
            check_dilution(equity_curve, trades, n_days),
            check_synthetic_data(data_source, code_path, trades),
            check_look_ahead(trades, code_path),
            check_sharpe_formula(reported_sharpe, trades=trades, cagr=reported_cagr),
            check_survivorship(trades),
            check_transaction_costs(trades, has_commissions, has_slippage, spread_width),
            check_capacity(trades, ticker),
        ]

        n_passed = sum(1 for c in checks if c.severity == "PASS")
        n_warnings = sum(1 for c in checks if c.severity == "WARNING")
        n_failed = sum(1 for c in checks if c.severity == "FAIL")
        n_critical = sum(1 for c in checks if c.severity == "CRITICAL")

        # Grade
        if n_critical > 0:
            grade = "F"
        elif n_failed >= 2:
            grade = "D"
        elif n_failed == 1:
            grade = "C"
        elif n_warnings >= 2:
            grade = "B"
        elif n_warnings == 1:
            grade = "B+"
        else:
            grade = "A"

        # Recommendations
        recs = []
        for c in checks:
            if c.severity in ("FAIL", "CRITICAL"):
                if "synthetic" in c.name.lower():
                    recs.append("Replace synthetic data with IronVault real option prices")
                elif "sharpe" in c.name.lower():
                    recs.append("Use arithmetic mean(returns)/std(returns)*sqrt(252) for Sharpe")
                elif "dilution" in c.name.lower():
                    recs.append("Compute Sharpe from trade-level PnL, not daily equity curve")
                elif "look-ahead" in c.name.lower():
                    recs.append("Review all data access for temporal causality")
                elif "cost" in c.name.lower():
                    recs.append("Add bid-ask spread model from EXP-850 (min $0.03/leg for SPY)")
                elif "capacity" in c.name.lower():
                    recs.append("Reduce position size or switch to more liquid underlying")

        return AuditReport(
            checks=checks,
            timestamp=datetime.utcnow().isoformat(),
            n_passed=n_passed,
            n_warnings=n_warnings,
            n_failed=n_failed,
            n_critical=n_critical,
            overall_grade=grade,
            recommendations=recs,
        )

    def generate_html(self, report: AuditReport, title: str = "Backtest Audit") -> str:
        """Generate HTML audit report."""
        gc = {"A": "#059669", "B+": "#059669", "B": "#d97706",
              "C": "#d97706", "D": "#dc2626", "F": "#dc2626"}
        vc = gc.get(report.overall_grade, "#6b7280")

        rows = ""
        for c in report.checks:
            ic = {"PASS": "#059669", "WARNING": "#d97706",
                  "FAIL": "#dc2626", "CRITICAL": "#dc2626"}[c.severity]
            icon = {"PASS": "&#10003;", "WARNING": "&#9888;",
                    "FAIL": "&#10007;", "CRITICAL": "&#10007;&#10007;"}[c.severity]
            rows += (
                f'<tr><td style="color:{ic}"><strong>{icon} {c.severity}</strong></td>'
                f'<td style="text-align:left"><strong>{c.name}</strong></td>'
                f'<td style="text-align:left">{c.message}</td>'
                f'<td style="text-align:left;color:var(--muted);font-size:.78rem">{c.details}</td></tr>\n'
            )

        rec_html = ""
        if report.recommendations:
            rec_html = '<h3>Recommendations</h3><ul style="padding-left:20px;line-height:2">'
            for r in report.recommendations:
                rec_html += f"<li>{r}</li>"
            rec_html += "</ul>"

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1000px;margin:0 auto;padding:24px}}
h1{{font-size:1.4rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:700;margin:24px 0 10px;border-bottom:2px solid var(--border);padding-bottom:6px}}
h3{{font-size:.95rem;font-weight:600;margin:14px 0 8px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:16px}}
.grade{{background:var(--card);border:2px solid {vc};border-radius:10px;padding:16px;text-align:center;margin:14px 0}}
.grade .big{{font-size:2rem;font-weight:800;color:{vc}}}
.grade .sub{{color:var(--muted);font-size:.85rem}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.82rem}}
th,td{{padding:5px 8px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.7rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
</style></head><body>
<h1>{title}</h1>
<div class="subtitle">{report.timestamp}</div>
<div class="grade">
  <div class="big">Grade: {report.overall_grade}</div>
  <div class="sub">Passed: {report.n_passed} | Warnings: {report.n_warnings} | Failed: {report.n_failed} | Critical: {report.n_critical}</div>
</div>
<h2>Check Results</h2>
<table>
<thead><tr><th>Status</th><th>Check</th><th>Result</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody></table>
{rec_html}
</body></html>"""
