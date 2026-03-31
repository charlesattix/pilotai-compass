"""
COMPASS auto-documentation generator.

Scans all ``compass/*.py`` modules, extracts docstrings, classes, functions,
and inter-module imports to produce a single self-contained HTML reference.

Usage::

    from compass.generate_docs import generate_docs
    generate_docs()  # → reports/compass_docs.html

    # or from CLI
    python -m compass.generate_docs
"""

from __future__ import annotations

import ast
import html as html_mod
import logging
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
COMPASS_DIR = ROOT / "compass"
DEFAULT_OUTPUT = ROOT / "reports" / "compass_docs.html"


# ── AST extraction ───────────────────────────────────────────────────────


@dataclass
class ParamInfo:
    name: str
    annotation: str
    default: str


@dataclass
class FuncInfo:
    name: str
    docstring: str
    params: List[ParamInfo]
    returns: str
    decorators: List[str]
    is_method: bool = False
    is_static: bool = False
    is_classmethod: bool = False


@dataclass
class ClassInfo:
    name: str
    docstring: str
    bases: List[str]
    methods: List[FuncInfo]
    init_params: List[ParamInfo]


@dataclass
class ModuleInfo:
    filename: str
    name: str              # module name without .py
    docstring: str
    classes: List[ClassInfo]
    functions: List[FuncInfo]  # module-level public functions
    imports_from: List[str]    # compass modules this module imports from
    constants: List[str]       # ALL_CAPS names at module level


def _get_annotation(node) -> str:
    """Best-effort annotation string from AST node."""
    if node is None:
        return ""
    return ast.unparse(node) if hasattr(ast, "unparse") else ""


def _get_default(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node) if hasattr(ast, "unparse") else repr(ast.literal_eval(node))
    except Exception:
        return "..."


def _extract_params(args: ast.arguments) -> List[ParamInfo]:
    """Extract parameter info from function arguments."""
    params: List[ParamInfo] = []
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg == "self":
            continue
        ann = _get_annotation(arg.annotation)
        default_idx = i - defaults_offset
        default = _get_default(args.defaults[default_idx]) if default_idx >= 0 else ""
        params.append(ParamInfo(arg.arg, ann, default))

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        ann = _get_annotation(arg.annotation)
        dflt = _get_default(default) if default else ""
        params.append(ParamInfo(arg.arg, ann, dflt))

    return params


def _extract_func(node: ast.FunctionDef) -> FuncInfo:
    decorators = []
    for d in node.decorator_list:
        decorators.append(ast.unparse(d) if hasattr(ast, "unparse") else "")
    return FuncInfo(
        name=node.name,
        docstring=ast.get_docstring(node) or "",
        params=_extract_params(node.args),
        returns=_get_annotation(node.returns),
        decorators=decorators,
        is_static="staticmethod" in decorators,
        is_classmethod="classmethod" in decorators,
    )


def _extract_class(node: ast.ClassDef) -> ClassInfo:
    bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
    methods: List[FuncInfo] = []
    init_params: List[ParamInfo] = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fi = _extract_func(item)
            fi.is_method = True
            if item.name == "__init__":
                init_params = fi.params
            elif not item.name.startswith("_"):
                methods.append(fi)

    return ClassInfo(
        name=node.name,
        docstring=ast.get_docstring(node) or "",
        bases=bases,
        methods=methods,
        init_params=init_params,
    )


def _extract_imports(tree: ast.Module) -> List[str]:
    """Extract compass.* imports."""
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod.startswith("compass."):
                submod = mod.split(".")[1]
                imports.add(submod)
            elif mod == "compass":
                imports.add("__init__")
    return sorted(imports)


def _extract_constants(tree: ast.Module) -> List[str]:
    """Extract ALL_CAPS module-level names."""
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                    constants.append(target.id)
    return constants


def parse_module(filepath: Path) -> ModuleInfo:
    """Parse a single Python module file."""
    source = filepath.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ModuleInfo(filepath.name, filepath.stem, "", [], [], [], [])

    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_extract_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(_extract_func(node))

    return ModuleInfo(
        filename=filepath.name,
        name=filepath.stem,
        docstring=ast.get_docstring(tree) or "",
        classes=classes,
        functions=functions,
        imports_from=_extract_imports(tree),
        constants=_extract_constants(tree),
    )


def scan_compass_modules(compass_dir: Path = COMPASS_DIR) -> List[ModuleInfo]:
    """Scan all compass/*.py modules."""
    modules = []
    for fp in sorted(compass_dir.glob("*.py")):
        if fp.name.startswith("__"):
            continue
        modules.append(parse_module(fp))
    return modules


# ── Dependency matrix ────────────────────────────────────────────────────


def build_dependency_matrix(modules: List[ModuleInfo]) -> Dict[str, List[str]]:
    """Build {module_name: [modules_it_imports_from]}."""
    return {m.name: m.imports_from for m in modules}


# ── Quick Start content ──────────────────────────────────────────────────

QUICK_START = {
    "Signal Model": {
        "module": "signal_model",
        "code": textwrap.dedent("""\
            from compass.signal_model import SignalModel
            model = SignalModel(model_dir="ml/models")
            stats = model.train(features_df, labels)
            result = model.predict(feature_dict)
            print(result["probability"], result["signal"])
        """),
    },
    "Ensemble Model": {
        "module": "ensemble_signal_model",
        "code": textwrap.dedent("""\
            from compass.ensemble_signal_model import EnsembleSignalModel
            model = EnsembleSignalModel(model_dir="ml/models")
            stats = model.train(features_df, labels, n_wf_folds=5)
            probas = model.predict_batch(features_df)
        """),
    },
    "Stress Testing": {
        "module": "stress_test",
        "code": textwrap.dedent("""\
            from compass.stress_test import StressTester
            from compass.crisis_hedge import CrisisHedgeConfig
            tester = StressTester(daily_returns, n_simulations=10000)
            results = tester.run_all(crisis_hedge_config=CrisisHedgeConfig())
            print(results["summary"]["risk_rating"])
        """),
    },
    "Risk Management": {
        "module": "risk_gate",
        "code": textwrap.dedent("""\
            from compass.risk_gate import RiskGate
            from compass.regime_gate import RegimeGate
            risk = RiskGate()
            gate = RegimeGate()
            decision = gate.evaluate("bull")
            if decision.should_trade:
                # ... proceed with position sizing
        """),
    },
    "Walk-Forward Validation": {
        "module": "walk_forward",
        "code": textwrap.dedent("""\
            from compass.walk_forward import validate_model
            results = validate_model(training_df)
            print(f"OOS AUC: {results['aggregate']['auc_mean']:.4f}")
        """),
    },
}


# ── HTML generation ──────────────────────────────────────────────────────


def _esc(text: str) -> str:
    return html_mod.escape(text)


def _sig(func: FuncInfo) -> str:
    """Format a function signature as HTML."""
    parts = []
    for p in func.params:
        s = _esc(p.name)
        if p.annotation:
            s += f": <em>{_esc(p.annotation)}</em>"
        if p.default:
            s += f" = {_esc(p.default)}"
        parts.append(s)
    ret = f" → <em>{_esc(func.returns)}</em>" if func.returns else ""
    prefix = ""
    if func.is_static:
        prefix = '<span class="tag">static</span> '
    elif func.is_classmethod:
        prefix = '<span class="tag">classmethod</span> '
    return f'{prefix}<strong>{_esc(func.name)}</strong>({", ".join(parts)}){ret}'


def generate_html(modules: List[ModuleInfo], dep_matrix: Dict[str, List[str]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_classes = sum(len(m.classes) for m in modules)
    n_funcs = sum(len(m.functions) for m in modules)
    n_methods = sum(sum(len(c.methods) for c in m.classes) for m in modules)

    # Module index
    index_items = "".join(
        f'<li><a href="#{m.name}">{m.name}</a>'
        f' — <small>{len(m.classes)} classes, {len(m.functions)} functions</small></li>\n'
        for m in modules
    )

    # Module sections
    module_sections = ""
    for m in modules:
        doc_first_line = m.docstring.split("\n")[0] if m.docstring else ""

        # Classes
        class_html = ""
        for cls in m.classes:
            init_sig = ""
            if cls.init_params:
                parts = []
                for p in cls.init_params:
                    s = _esc(p.name)
                    if p.annotation:
                        s += f": <em>{_esc(p.annotation)}</em>"
                    if p.default:
                        s += f" = {_esc(p.default)}"
                    parts.append(s)
                init_sig = ", ".join(parts)

            method_rows = ""
            for meth in cls.methods:
                doc_line = meth.docstring.split("\n")[0] if meth.docstring else ""
                method_rows += (
                    f'<tr><td>{_sig(meth)}</td>'
                    f'<td class="doc-cell">{_esc(doc_line)}</td></tr>\n'
                )

            bases_str = f' <small>({", ".join(_esc(b) for b in cls.bases)})</small>' if cls.bases else ""
            cls_doc = _esc(cls.docstring.split("\n")[0]) if cls.docstring else ""

            class_html += f"""
            <div class="class-block">
              <h4>class {_esc(cls.name)}{bases_str}</h4>
              <p class="doc">{cls_doc}</p>
              {f'<p class="sig"><code>{_esc(cls.name)}({init_sig})</code></p>' if init_sig else ''}
              {'<table class="methods"><thead><tr><th>Method</th><th>Description</th></tr></thead><tbody>' + method_rows + '</tbody></table>' if method_rows else ''}
            </div>"""

        # Functions
        func_html = ""
        for func in m.functions:
            doc_line = func.docstring.split("\n")[0] if func.docstring else ""
            func_html += f'<div class="func-block"><p>{_sig(func)}</p><p class="doc">{_esc(doc_line)}</p></div>\n'

        # Dependencies
        deps = dep_matrix.get(m.name, [])
        dep_html = ", ".join(f'<a href="#{d}">{d}</a>' for d in deps) if deps else "<em>none</em>"

        # Constants
        const_html = ", ".join(f'<code>{_esc(c)}</code>' for c in m.constants[:10]) if m.constants else ""

        module_sections += f"""
        <div class="module" id="{m.name}">
          <h3>{_esc(m.filename)}</h3>
          <p class="doc">{_esc(doc_first_line)}</p>
          <p class="dep"><strong>Imports from:</strong> {dep_html}</p>
          {f'<p class="dep"><strong>Constants:</strong> {const_html}</p>' if const_html else ''}
          {class_html}
          {f'<h4>Module Functions</h4>{func_html}' if func_html else ''}
        </div>
        <hr>"""

    # Quick Start
    qs_html = ""
    for title, info in QUICK_START.items():
        qs_html += f"""
        <div class="qs-block">
          <h4>{_esc(title)} <small>({info['module']}.py)</small></h4>
          <pre><code>{_esc(info['code'])}</code></pre>
        </div>"""

    # Dependency matrix table
    all_names = sorted(set(m.name for m in modules))
    dep_header = "".join(f'<th class="rot"><div>{n[:12]}</div></th>' for n in all_names)
    dep_rows = ""
    for m in modules:
        if m.name not in all_names:
            continue
        cells = ""
        for target in all_names:
            has = target in dep_matrix.get(m.name, [])
            cells += f'<td class="{"dep-yes" if has else ""}">{("●" if has else "")}</td>'
        dep_rows += f'<tr><td class="dep-label">{m.name[:20]}</td>{cells}</tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>COMPASS Documentation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.2em; }}
  h3 {{ color: #1e40af; margin-top: 2em; }}
  h4 {{ color: #334155; margin-top: 1.2em; margin-bottom: 0.3em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 2em; }}
  .doc {{ color: #475569; font-size: 0.9em; margin: 0.2em 0 0.8em; }}
  .sig {{ font-size: 0.85em; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; }}
  .dep {{ font-size: 0.85em; color: #64748b; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font-size: 0.88em; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
  pre {{ background: #1e293b; color: #e2e8f0; padding: 1em; border-radius: 6px;
         overflow-x: auto; font-size: 0.85em; }}
  pre code {{ background: none; padding: 0; }}
  em {{ color: #6366f1; font-style: normal; }}
  .tag {{ background: #dbeafe; color: #1e40af; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; }}
  .module {{ margin: 0.5em 0; }}
  .class-block {{ background: #fff; border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;
                  border-radius: 4px; padding: 1em; margin: 0.8em 0; }}
  .func-block {{ padding: 0.3em 0; }}
  .methods {{ width: 100%; font-size: 0.85em; border-collapse: collapse; margin: 0.5em 0; }}
  .methods th {{ background: #f8fafc; text-align: left; padding: 4px 8px; border-bottom: 1px solid #e2e8f0; }}
  .methods td {{ padding: 4px 8px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
  .doc-cell {{ color: #64748b; font-size: 0.92em; }}
  .qs-block {{ margin: 1em 0; }}
  .index {{ column-count: 3; font-size: 0.9em; }}
  @media (max-width: 800px) {{ .index {{ column-count: 1; }} }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 1.5em 0; }}
  .dep-matrix {{ font-size: 0.7em; border-collapse: collapse; }}
  .dep-matrix th, .dep-matrix td {{ border: 1px solid #e2e8f0; padding: 2px 4px; text-align: center; }}
  .dep-matrix .dep-label {{ text-align: left; font-weight: 500; white-space: nowrap; }}
  .dep-yes {{ background: #dbeafe; color: #1e40af; }}
  .rot {{ writing-mode: vertical-rl; transform: rotate(180deg); max-width: 20px; }}
  .rot div {{ max-height: 80px; overflow: hidden; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>COMPASS System Documentation</h1>
<div class="meta">
  {len(modules)} modules &middot; {n_classes} classes &middot;
  {n_funcs} module functions &middot; {n_methods} methods &middot;
  Generated {now}
</div>

<h2>Quick Start</h2>
{qs_html}

<h2>Module Index</h2>
<ul class="index">
{index_items}
</ul>

<h2>Module Reference</h2>
{module_sections}

<h2>Dependency Matrix</h2>
<p class="doc">● = row module imports from column module</p>
<div style="overflow-x:auto">
<table class="dep-matrix">
<thead><tr><th></th>{dep_header}</tr></thead>
<tbody>{dep_rows}</tbody>
</table>
</div>

<footer>Generated by <code>compass/generate_docs.py</code></footer>
</body></html>"""
    return html


# ── Public API ───────────────────────────────────────────────────────────


def generate_docs(
    compass_dir: str = str(COMPASS_DIR),
    output: str = str(DEFAULT_OUTPUT),
) -> str:
    """Scan compass modules and generate HTML documentation.

    Returns absolute path to the generated file.
    """
    logger.info("Scanning %s", compass_dir)
    modules = scan_compass_modules(Path(compass_dir))
    logger.info("Found %d modules", len(modules))

    dep_matrix = build_dependency_matrix(modules)
    html = generate_html(modules, dep_matrix)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Docs written to %s (%d bytes)", out, len(html))
    return str(out.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = generate_docs()
    print(f"Documentation: {path}")
