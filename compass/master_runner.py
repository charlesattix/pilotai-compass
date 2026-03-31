"""
Master orchestration engine — runs the full COMPASS pipeline.

12-stage configurable pipeline:
  1. Pipeline validation     6. Model prediction     10. Hedge overlay
  2. Deploy readiness        7. Position sizing      11. P&L calculation
  3. Data pipeline           8. Risk limits          12. Report generation
  4. Feature computation     9. Order generation
  5. Signal generation

Supports: YAML-style config, dry-run mode, graceful skip on error,
per-stage timing, comprehensive HTML run report.

All methods operate on pre-loaded data — no broker connections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


@dataclass
class StageResult:
    name: str
    status: StageStatus
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None


@dataclass
class RunConfig:
    """Pipeline configuration."""
    experiments: List[str] = field(default_factory=lambda: ["EXP-400"])
    enabled_stages: List[str] = field(default_factory=lambda: [
        "validate", "readiness", "data", "features", "signals",
        "model", "sizing", "risk", "orders", "hedge", "pnl", "reports",
    ])
    dry_run: bool = False
    account_size: float = 100000
    data: Optional[Dict[str, Any]] = None


@dataclass
class MasterRunResult:
    stages: List[StageResult]
    total_duration_ms: float
    n_success: int
    n_failed: int
    n_skipped: int
    config: RunConfig


STAGE_NAMES = [
    "validate", "readiness", "data", "features", "signals",
    "model", "sizing", "risk", "orders", "hedge", "pnl", "reports",
]


class MasterRunner:
    """Master pipeline orchestrator.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Optional[RunConfig] = None) -> None:
        self.config = config or RunConfig()
        self._results: List[StageResult] = []
        self._context: Dict[str, Any] = {}
        self._stage_fns: Dict[str, Callable] = {
            "validate": self._stage_validate,
            "readiness": self._stage_readiness,
            "data": self._stage_data,
            "features": self._stage_features,
            "signals": self._stage_signals,
            "model": self._stage_model,
            "sizing": self._stage_sizing,
            "risk": self._stage_risk,
            "orders": self._stage_orders,
            "hedge": self._stage_hedge,
            "pnl": self._stage_pnl,
            "reports": self._stage_reports,
        }

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_validate(self) -> Dict:
        """Check module imports and basic health."""
        checks = {}
        for mod in ["compass.regime", "compass.vol_forecaster",
                      "compass.risk_orchestrator", "compass.portfolio_constructor"]:
            try:
                __import__(mod)
                checks[mod] = True
            except ImportError:
                checks[mod] = False
        if not all(checks.values()):
            raise RuntimeError(f"Validation failed: {checks}")
        return checks

    def _stage_readiness(self) -> Dict:
        """Check data availability and config."""
        ready = {
            "config_valid": len(self.config.experiments) > 0,
            "account_size_valid": self.config.account_size > 0,
            "stages_configured": len(self.config.enabled_stages) > 0,
        }
        if not all(ready.values()):
            raise RuntimeError(f"Readiness failed: {ready}")
        return ready

    def _stage_data(self) -> Dict:
        """Load or generate market data."""
        data = self.config.data or {}
        if "prices" not in data:
            rng = np.random.default_rng(42)
            n = 252
            idx = pd.bdate_range("2024-01-02", periods=n)
            prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
            data["prices"] = prices
            data["returns"] = prices.pct_change().dropna()
            data["volume"] = pd.Series(rng.integers(1e6, 5e6, n).astype(float), index=idx)
        self._context["data"] = data
        return {"n_days": len(data["prices"]), "symbols": self.config.experiments}

    def _stage_features(self) -> Dict:
        """Compute features from market data."""
        data = self._context.get("data", {})
        prices = data.get("prices", pd.Series(dtype=float))
        returns = data.get("returns", prices.pct_change().dropna())
        features = {
            "volatility": returns.rolling(21).std().fillna(0) * np.sqrt(252),
            "momentum": prices.pct_change(20).fillna(0),
            "volume_ma": data.get("volume", pd.Series(dtype=float)).rolling(20).mean().fillna(0),
        }
        self._context["features"] = features
        return {"n_features": len(features)}

    def _stage_signals(self) -> Dict:
        """Generate trading signals."""
        feat = self._context.get("features", {})
        mom = feat.get("momentum", pd.Series(dtype=float))
        signal = mom.apply(lambda x: 1.0 if x > 0.01 else (-1.0 if x < -0.01 else 0.0))
        self._context["signal"] = signal
        return {"n_signals": int((signal != 0).sum())}

    def _stage_model(self) -> Dict:
        """Run model predictions (mock)."""
        signal = self._context.get("signal", pd.Series(dtype=float))
        confidence = signal.abs().clip(0, 1)
        self._context["confidence"] = confidence
        return {"avg_confidence": float(confidence.mean())}

    def _stage_sizing(self) -> Dict:
        """Determine position sizes."""
        signal = self._context.get("signal", pd.Series(dtype=float))
        conf = self._context.get("confidence", pd.Series(1.0, index=signal.index))
        size = signal * conf * self.config.account_size * 0.02
        self._context["sizes"] = size
        return {"max_size": float(size.abs().max())}

    def _stage_risk(self) -> Dict:
        """Apply risk limits."""
        sizes = self._context.get("sizes", pd.Series(dtype=float))
        max_size = self.config.account_size * 0.05
        clipped = sizes.clip(-max_size, max_size)
        self._context["approved_sizes"] = clipped
        n_clipped = int((sizes.abs() > max_size).sum())
        return {"n_clipped": n_clipped, "max_allowed": max_size}

    def _stage_orders(self) -> Dict:
        """Generate orders from approved sizes."""
        sizes = self._context.get("approved_sizes", pd.Series(dtype=float))
        orders = sizes[sizes != 0]
        self._context["orders"] = orders
        return {"n_orders": len(orders)}

    def _stage_hedge(self) -> Dict:
        """Apply hedge overlay."""
        data = self._context.get("data", {})
        prices = data.get("prices", pd.Series(dtype=float))
        if prices.empty:
            return {"hedge_ratio": 0.0}
        hwm = prices.expanding().max()
        dd = float((1 - prices / hwm).iloc[-1])
        hedge_ratio = min(dd * 2, 0.5)
        self._context["hedge_ratio"] = hedge_ratio
        return {"drawdown": dd, "hedge_ratio": hedge_ratio}

    def _stage_pnl(self) -> Dict:
        """Calculate P&L."""
        data = self._context.get("data", {})
        returns = data.get("returns", pd.Series(dtype=float))
        signal = self._context.get("signal", pd.Series(dtype=float))
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        if aligned.empty:
            return {"pnl": 0.0, "sharpe": 0.0}
        strat_ret = aligned["sig"].shift(1).fillna(0) * aligned["ret"]
        mu = float(strat_ret.mean())
        std = float(strat_ret.std())
        sharpe = mu / std * np.sqrt(252) if std > 1e-12 else 0.0
        pnl = float((1 + strat_ret).prod() - 1)
        self._context["pnl"] = pnl
        self._context["sharpe"] = sharpe
        return {"pnl": pnl, "sharpe": sharpe}

    def _stage_reports(self) -> Dict:
        """Generate summary report data."""
        return {
            "pnl": self._context.get("pnl", 0.0),
            "sharpe": self._context.get("sharpe", 0.0),
            "n_stages": len(self._results),
            "experiments": self.config.experiments,
        }

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    def _run_stage(self, name: str) -> StageResult:
        if name not in self.config.enabled_stages:
            return StageResult(name, StageStatus.SKIPPED)

        if self.config.dry_run:
            return StageResult(name, StageStatus.DRY_RUN, 0.0)

        fn = self._stage_fns.get(name)
        if fn is None:
            return StageResult(name, StageStatus.SKIPPED, error="No implementation")

        t0 = time.perf_counter()
        try:
            output = fn()
            ms = (time.perf_counter() - t0) * 1000
            return StageResult(name, StageStatus.SUCCESS, ms, output)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            logger.warning("Stage '%s' failed: %s", name, e)
            return StageResult(name, StageStatus.FAILED, ms, error=str(e))

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> MasterRunResult:
        """Execute all pipeline stages."""
        self._results.clear()
        self._context.clear()
        t0 = time.perf_counter()

        for name in STAGE_NAMES:
            result = self._run_stage(name)
            self._results.append(result)

        total_ms = (time.perf_counter() - t0) * 1000
        return MasterRunResult(
            stages=list(self._results),
            total_duration_ms=total_ms,
            n_success=sum(1 for r in self._results if r.status == StageStatus.SUCCESS),
            n_failed=sum(1 for r in self._results if r.status == StageStatus.FAILED),
            n_skipped=sum(1 for r in self._results if r.status in (StageStatus.SKIPPED, StageStatus.DRY_RUN)),
            config=self.config,
        )

    @property
    def results(self) -> List[StageResult]:
        return list(self._results)

    @property
    def context(self) -> Dict[str, Any]:
        return dict(self._context)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, result: MasterRunResult,
        output_path: str = "reports/master_run.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        colors = {"success": "#27ae60", "failed": "#e74c3c",
                    "skipped": "#999", "dry_run": "#2980b9"}

        # Timing waterfall SVG
        max_ms = max((r.duration_ms for r in result.stages), default=1) or 1
        n = len(result.stages)
        sw, sh = 720, n * 34 + 30
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sw}" height="{sh}" '
               f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        for i, s in enumerate(result.stages):
            y = 10 + i * 34
            c = colors.get(s.status.value, "#999")
            bw = max(s.duration_ms / max_ms * 350, 4)
            svg.append(f'<text x="148" y="{y + 20}" text-anchor="end" font-size="11" fill="#333">{s.name}</text>')
            svg.append(f'<rect x="155" y="{y + 5}" width="{bw:.0f}" height="22" fill="{c}" rx="3"/>')
            svg.append(f'<text x="{160 + bw:.0f}" y="{y + 20}" font-size="10" fill="#666">{s.duration_ms:.1f}ms</text>')
        svg.append("</svg>")
        waterfall = "\n".join(svg)

        rows = [
            f"<tr><td style='text-align:left'>{s.name}</td>"
            f"<td style='color:{colors.get(s.status.value, '#999')};font-weight:bold'>"
            f"{s.status.value.upper()}</td>"
            f"<td>{s.duration_ms:.1f}</td>"
            f"<td style='text-align:left'>{s.error or ''}</td></tr>"
            for s in result.stages
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Master Run</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.big {{ font-size: 1.5em; font-weight: bold; }}
</style></head><body>
<h1>Master Pipeline Run</h1>
<div class="summary">
<p class="big" style="color:{'#27ae60' if result.n_failed == 0 else '#e74c3c'}">
{result.n_success}/{len(result.stages)} stages passed</p>
<p>Total: {result.total_duration_ms:.0f}ms | Failed: {result.n_failed} | Skipped: {result.n_skipped}
| Dry run: {result.config.dry_run} | Experiments: {', '.join(result.config.experiments)}</p>
</div>
<h2>Timing Waterfall</h2>
{waterfall}
<h2>Stage Details</h2>
<table><tr><th style='text-align:left'>Stage</th><th>Status</th><th>Duration (ms)</th>
<th style='text-align:left'>Error</th></tr>
{''.join(rows)}</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
