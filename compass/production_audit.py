"""Production readiness auditor for all compass modules.

Scans every .py in compass/, evaluates code quality, test coverage,
data dependencies, latency, capacity, and external deps. Ranks
modules by production-readiness score.

Pure-Python — no external dependencies.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

COMPASS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(COMPASS_DIR)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModuleAudit:
    """Audit result for one module."""
    name: str
    path: str
    lines: int
    n_functions: int
    n_classes: int
    has_docstring: bool
    has_tests: bool
    imports_ok: bool
    import_error: str
    external_deps: List[str]       # non-stdlib, non-compass imports
    data_deps: List[str]           # detected data requirements
    estimated_latency: str         # "fast" (<100ms), "medium" (<1s), "slow" (>1s)
    quality_score: float           # 0-10
    production_ready: bool
    blockers: List[str]
    category: str


@dataclass
class AuditReport:
    """Full audit output."""
    n_modules: int
    n_production_ready: int
    n_with_tests: int
    n_import_ok: int
    modules: List[ModuleAudit]
    top_10: List[ModuleAudit]
    categories: Dict[str, int]
    avg_quality: float


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

def discover_modules() -> List[str]:
    """Find all .py files in compass/."""
    mods = []
    for f in sorted(os.listdir(COMPASS_DIR)):
        if f.endswith(".py") and f != "__init__.py" and f != "production_audit.py":
            mods.append(f[:-3])
    return mods


def discover_tests() -> Set[str]:
    """Find all module names that have test files."""
    names: Set[str] = set()
    tests_dir = os.path.join(PROJECT_ROOT, "tests")
    compass_tests = os.path.join(COMPASS_DIR, "tests")
    for d in [tests_dir, compass_tests]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("test_") and f.endswith(".py"):
                names.add(f[5:-3])
    return names


# ---------------------------------------------------------------------------
# Module analysis
# ---------------------------------------------------------------------------

# Known external packages (not stdlib, not compass)
EXTERNAL_PACKAGES = {
    "numpy", "np", "pandas", "pd", "sklearn", "scipy", "xgboost",
    "lightgbm", "torch", "tensorflow", "requests", "alpaca",
    "polygon", "yfinance", "plotly", "matplotlib",
}

# Category keywords
CATEGORIES = {
    "signal": ["signal", "alpha", "sentiment", "momentum", "indicator", "feature"],
    "regime": ["regime", "hmm", "markov"],
    "risk": ["risk", "hedge", "crisis", "stress", "drawdown", "stop", "tail"],
    "execution": ["execution", "order", "fill", "slippage", "routing"],
    "ml": ["model", "ensemble", "train", "retrain", "xgboost", "learner"],
    "portfolio": ["portfolio", "position", "sizing", "allocation", "weight"],
    "backtest": ["backtest", "walk_forward", "validation", "benchmark"],
    "data": ["data", "collect", "pipeline", "cache", "provider"],
    "monitoring": ["monitor", "alert", "health", "audit", "report", "dashboard"],
    "analysis": ["analysis", "analyzer", "decay", "correlation", "cluster"],
}

# Data dependency keywords
DATA_KEYWORDS = {
    "options_cache": "Options chain database (options_cache.db)",
    "polygon": "Polygon.io real-time market data API",
    "alpaca": "Alpaca trading API",
    "yfinance": "Yahoo Finance historical data",
    "vix": "VIX index data (real-time)",
    "treasury": "Treasury yield data",
    "earnings": "Earnings calendar data",
    "credit_spread": "Credit spread indices (HYG/TLT)",
}


def _categorise(name: str, source: str) -> str:
    name_lower = name.lower()
    source_lower = source.lower()[:500]
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in name_lower or kw in source_lower:
                return cat
    return "other"


def _detect_external_deps(source: str) -> List[str]:
    """Find imports of external (non-stdlib, non-compass) packages."""
    deps: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return deps
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                if pkg in EXTERNAL_PACKAGES:
                    deps.append(pkg)
        elif isinstance(node, ast.ImportFrom) and node.module:
            pkg = node.module.split(".")[0]
            if pkg in EXTERNAL_PACKAGES:
                deps.append(pkg)
    return list(set(deps))


def _detect_data_deps(source: str) -> List[str]:
    """Detect data requirements from source code."""
    deps: List[str] = []
    source_lower = source.lower()
    for keyword, desc in DATA_KEYWORDS.items():
        if keyword in source_lower:
            deps.append(desc)
    return deps


def _estimate_latency(lines: int, external: List[str], source: str) -> str:
    """Estimate execution latency."""
    has_ml = any(p in external for p in ["sklearn", "xgboost", "lightgbm", "torch"])
    has_heavy_loop = "for _ in range(50" in source or "for _ in range(10" in source
    if has_ml or has_heavy_loop:
        return "slow"
    if lines > 500 or external:
        return "medium"
    return "fast"


def _compute_quality(
    lines: int, n_func: int, n_cls: int, has_doc: bool,
    has_tests: bool, imports_ok: bool, external: List[str],
) -> Tuple[float, List[str]]:
    """Compute quality score 0-10 and identify blockers."""
    score = 5.0  # base
    blockers: List[str] = []

    if has_doc: score += 1.0
    else: blockers.append("Missing module docstring")

    if has_tests: score += 2.0
    else: blockers.append("No test file found")

    if imports_ok: score += 1.0
    else: blockers.append("Import fails"); score -= 2.0

    if not external: score += 0.5
    else: blockers.append(f"External deps: {', '.join(external)}")

    if n_func >= 3: score += 0.3
    if n_cls >= 1: score += 0.2

    if lines > 1000: score -= 0.5  # too large
    if lines < 20: score -= 1.0; blockers.append("Module too small (<20 lines)")

    return min(10.0, max(0.0, round(score, 1))), blockers


def audit_module(name: str, test_names: Set[str]) -> ModuleAudit:
    """Audit one compass module."""
    path = os.path.join(COMPASS_DIR, f"{name}.py")
    try:
        with open(path) as f:
            source = f.read()
    except Exception:
        return ModuleAudit(name, path, 0, 0, 0, False, False, False,
                           "Cannot read file", [], [], "unknown", 0, False, ["Cannot read"], "other")

    lines = len(source.split("\n"))

    # AST analysis
    n_func = n_cls = 0
    has_doc = False
    try:
        tree = ast.parse(source)
        n_func = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        n_cls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        has_doc = bool(tree.body and isinstance(tree.body[0], ast.Expr)
                       and isinstance(getattr(tree.body[0], "value", None), ast.Constant))
    except SyntaxError:
        pass

    # Test coverage
    has_tests = name in test_names
    # Check name variants
    for variant in [name.replace("_v2", ""), name.replace("_analyzer", ""),
                    name.split("_")[0], name + "_v2"]:
        if variant in test_names:
            has_tests = True

    # Import check
    imports_ok = True
    import_error = ""
    try:
        full = f"compass.{name}"
        if full not in sys.modules:
            importlib.import_module(full)
    except Exception as e:
        imports_ok = False
        import_error = f"{type(e).__name__}: {str(e)[:100]}"

    external = _detect_external_deps(source)
    data_deps = _detect_data_deps(source)
    latency = _estimate_latency(lines, external, source)
    category = _categorise(name, source)

    quality, blockers = _compute_quality(lines, n_func, n_cls, has_doc,
                                          has_tests, imports_ok, external)
    prod_ready = quality >= 7.0 and imports_ok and has_tests

    return ModuleAudit(
        name, path, lines, n_func, n_cls, has_doc, has_tests,
        imports_ok, import_error, external, data_deps, latency,
        round(quality, 1), prod_ready, blockers, category,
    )


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def run_audit() -> AuditReport:
    """Audit all compass modules."""
    modules = discover_modules()
    test_names = discover_tests()
    audits: List[ModuleAudit] = []

    for name in modules:
        audits.append(audit_module(name, test_names))

    # Sort by quality descending
    audits.sort(key=lambda a: (-a.quality_score, a.name))

    top10 = [a for a in audits if a.production_ready][:10]
    n_ready = sum(1 for a in audits if a.production_ready)
    n_tests = sum(1 for a in audits if a.has_tests)
    n_import = sum(1 for a in audits if a.imports_ok)

    cats: Dict[str, int] = {}
    for a in audits:
        cats[a.category] = cats.get(a.category, 0) + 1

    qualities = [a.quality_score for a in audits]
    avg_q = sum(qualities) / len(qualities) if qualities else 0

    return AuditReport(
        len(audits), n_ready, n_tests, n_import,
        audits, top10, cats, round(avg_q, 1),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(report: AuditReport) -> str:
    """Generate markdown production readiness report."""
    lines: List[str] = []
    lines.append("# Production Readiness Audit Report")
    lines.append("")
    lines.append(f"**Modules scanned:** {report.n_modules}")
    lines.append(f"**Production ready:** {report.n_production_ready} ({report.n_production_ready/report.n_modules*100:.0f}%)")
    lines.append(f"**With tests:** {report.n_with_tests}")
    lines.append(f"**Import OK:** {report.n_import_ok}")
    lines.append(f"**Average quality:** {report.avg_quality}/10")
    lines.append("")

    # Top 10
    lines.append("## Top 10 Production-Ready Strategies")
    lines.append("")
    lines.append("| Rank | Module | Quality | Lines | Tests | Latency | Ext Deps | Category |")
    lines.append("|------|--------|---------|-------|-------|---------|----------|----------|")
    for i, m in enumerate(report.top_10, 1):
        deps = ", ".join(m.external_deps) if m.external_deps else "none"
        lines.append(f"| {i} | `{m.name}` | **{m.quality_score}** | {m.lines} | {'YES' if m.has_tests else 'no'} | {m.estimated_latency} | {deps} | {m.category} |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Modules |")
    lines.append("|----------|---------|")
    for cat in sorted(report.categories, key=lambda c: -report.categories[c]):
        lines.append(f"| {cat} | {report.categories[cat]} |")
    lines.append("")

    # Full table (top 50)
    lines.append("## Full Module Ranking (Top 50)")
    lines.append("")
    lines.append("| # | Module | Score | Lines | Tests | Import | Latency | Deps | Ready | Blockers |")
    lines.append("|---|--------|-------|-------|-------|--------|---------|------|-------|----------|")
    for i, m in enumerate(report.modules[:50], 1):
        deps = len(m.external_deps)
        blockers = "; ".join(m.blockers[:2]) if m.blockers else "—"
        ready = "YES" if m.production_ready else "no"
        imp = "OK" if m.imports_ok else "FAIL"
        lines.append(f"| {i} | `{m.name}` | {m.quality_score} | {m.lines} | "
                     f"{'YES' if m.has_tests else 'no'} | {imp} | {m.estimated_latency} | "
                     f"{deps} | {ready} | {blockers} |")
    lines.append("")

    # Import failures
    failures = [m for m in report.modules if not m.imports_ok]
    if failures:
        lines.append("## Import Failures")
        lines.append("")
        for m in failures[:20]:
            lines.append(f"- `{m.name}`: {m.import_error}")
        lines.append("")

    return "\n".join(lines)
