"""EXP-1950 — Adaptive Kelly position sizing for EXP-1220.

Hypothesis: dynamic Kelly-fraction sizing based on rolling edge/odds
and regime should lift Sharpe by +0.5 over a static 2× leverage
baseline without increasing drawdown.

Method:
  1. Load the canonical load_exp1220_dynamic stream (real Yahoo SPY +
     ^VIX + ^VIX3M, TailRiskProtector dynamic leverage). This is the
     same stream used by EXP-1860/1870 and the v4 audit.
  2. At each day t, look back 60/90/120 days and estimate
         mu  = rolling mean daily return
         var = rolling daily variance
     Continuous Kelly: f* = mu / var  (in units of portfolio stake)
  3. Apply fractional Kelly: f = min(kelly_frac * f*, max_leverage).
     Kelly fraction candidates: 0.25, 0.50, 0.75.
  4. Regime adjustment multiplier (compass.regime.RegimeClassifier):
         BULL     ×1.00
         LOW_VOL  ×1.10
         HIGH_VOL ×0.60
         BEAR     ×0.40
         CRASH    ×0.20
  5. Walk-forward 2020-2025 with rolling re-estimation.
  6. Baseline: static 2× leverage applied every day (the EXP-1860
     8/5/7.5/7.5 portfolio's EXP-1220 sleeve, but pure on this one
     stream for a clean A/B).

Success criterion: +0.5 Sharpe over baseline. KILL if <+0.2.

Rule Zero: only real Yahoo data. No synthetic returns, no model
proxies. The rolling estimator shifts by 1 day to avoid lookahead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics
from compass.regime import Regime, RegimeClassifier

REPORT_JSON = ROOT / "compass" / "reports" / "exp1950_adaptive_kelly.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1950_adaptive_kelly.html"

START = "2020-01-01"
END = "2025-12-31"

KELLY_FRACTIONS = [0.25, 0.50, 0.75]
LOOKBACKS = [60, 90, 120]
MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.0
BASELINE_LEV = 2.0

REGIME_MULT = {
    Regime.BULL.value:     1.00,
    Regime.LOW_VOL.value:  1.10,
    Regime.HIGH_VOL.value: 0.60,
    Regime.BEAR.value:     0.40,
    Regime.CRASH.value:    0.20,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220() -> pd.Series:
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    s = load_exp1220_dynamic()
    s.index = pd.DatetimeIndex(s.index)
    s = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END))]
    s.name = "exp1220"
    return s


def load_regime_series(index: pd.DatetimeIndex) -> pd.Series:
    from scripts.ultimate_portfolio import _fetch
    spy = _fetch("SPY", "2018-01-01", "2026-01-01")
    vix_df = _fetch("^VIX", "2018-01-01", "2026-01-01")
    vix = vix_df["Close"].squeeze()
    classifier = RegimeClassifier(trend_window=50, trend_threshold=5.0)
    regimes = classifier.classify_series(spy, vix)
    regimes = regimes.reindex(index, method="ffill").fillna(Regime.BULL).astype(str)
    return regimes


# ═══════════════════════════════════════════════════════════════════════════
# Kelly sizing
# ═══════════════════════════════════════════════════════════════════════════

def rolling_kelly(
    returns: pd.Series,
    lookback: int,
    kelly_frac: float,
    regime: Optional[pd.Series] = None,
    apply_regime: bool = True,
) -> pd.Series:
    """Per-day Kelly leverage (shifted by 1 day — no look-ahead).

    f* = mu / var, then scaled by kelly_frac, then scaled by the
    regime multiplier. Clipped to [MIN_LEVERAGE, MAX_LEVERAGE].
    """
    mu = returns.rolling(lookback, min_periods=20).mean()
    var = returns.rolling(lookback, min_periods=20).var()

    raw_kelly = (mu / var.replace(0, np.nan)).fillna(0.0)
    lev = (raw_kelly * kelly_frac).clip(MIN_LEVERAGE, MAX_LEVERAGE)

    # Shift by 1 so the leverage used on day t uses data through t-1
    lev = lev.shift(1).fillna(0.0)

    if apply_regime and regime is not None:
        mult = regime.map(REGIME_MULT).astype(float).fillna(1.0)
        lev = lev * mult
        lev = lev.clip(MIN_LEVERAGE, MAX_LEVERAGE)

    return lev


# ═══════════════════════════════════════════════════════════════════════════
# Baselines + variants
# ═══════════════════════════════════════════════════════════════════════════

def run_static(returns: pd.Series, leverage: float = BASELINE_LEV) -> Dict:
    """Constant leverage baseline."""
    rets = returns * leverage
    m = full_metrics(rets.values)
    return {
        "name": f"static_{leverage}x",
        "leverage": leverage,
        "metrics": m,
        "daily_returns": rets,
        "avg_leverage": float(leverage),
    }


def run_kelly(
    returns: pd.Series,
    regime: pd.Series,
    lookback: int,
    kelly_frac: float,
    apply_regime: bool,
) -> Dict:
    lev = rolling_kelly(returns, lookback, kelly_frac, regime, apply_regime)
    rets = returns * lev
    m = full_metrics(rets.values)
    return {
        "name": (f"kelly{kelly_frac}_lb{lookback}"
                 + ("_regime" if apply_regime else "_noregime")),
        "lookback": lookback,
        "kelly_fraction": kelly_frac,
        "regime_applied": apply_regime,
        "metrics": m,
        "avg_leverage": float(lev.mean()),
        "max_leverage": float(lev.max()),
        "min_leverage": float(lev.min()),
        "daily_returns": rets,
    }


def yearly(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("=" * 72)
    print("EXP-1950 — Adaptive Kelly Position Sizing")
    print("=" * 72)

    print("\n[load] EXP-1220 dynamic stream (real Yahoo)...")
    exp1220 = load_exp1220()
    base_m = full_metrics(exp1220.values)
    print(f"       {len(exp1220)} days  "
          f"CAGR {base_m['cagr_pct']:+.1f}%  "
          f"Sharpe {base_m['sharpe']:.2f}  "
          f"DD {base_m['max_dd_pct']:.1f}%")

    print("\n[load] regime series...")
    regimes = load_regime_series(exp1220.index)
    dist = regimes.value_counts(normalize=True).to_dict()
    print(f"       {dict((k, round(v, 3)) for k, v in dist.items())}")

    # ── Baselines ───────────────────────────────────────────────────
    print("\n[baseline] static 2× leverage")
    baseline = run_static(exp1220, BASELINE_LEV)
    bm = baseline["metrics"]
    print(f"  CAGR {bm['cagr_pct']:+7.1f}%  Sharpe {bm['sharpe']:.2f}  "
          f"DD {bm['max_dd_pct']:.1f}%  Calmar {bm['calmar']:.2f}")
    baseline_sharpe = bm["sharpe"]

    # ── Variants ────────────────────────────────────────────────────
    print("\n[variants] sweeping Kelly fraction × lookback × regime")
    variants: List[Dict] = []
    for kf in KELLY_FRACTIONS:
        for lb in LOOKBACKS:
            for reg in (False, True):
                v = run_kelly(exp1220, regimes, lb, kf, reg)
                variants.append(v)
                m = v["metrics"]
                lift = m["sharpe"] - baseline_sharpe
                flag = ""
                if lift >= 0.5:
                    flag = "★"
                elif lift >= 0.2:
                    flag = "+"
                elif lift < 0:
                    flag = "-"
                print(f"  {flag} {v['name']:32s} "
                      f"CAGR {m['cagr_pct']:+7.1f}%  "
                      f"Sharpe {m['sharpe']:5.2f} (Δ{lift:+.2f})  "
                      f"DD {m['max_dd_pct']:5.1f}%  "
                      f"avg_lev {v['avg_leverage']:.2f}")

    # ── Best variant ────────────────────────────────────────────────
    best = max(variants, key=lambda v: v["metrics"]["sharpe"])
    bestm = best["metrics"]
    lift = bestm["sharpe"] - baseline_sharpe

    print(f"\n[best] {best['name']}")
    print(f"  CAGR {bestm['cagr_pct']:+.1f}% (baseline {bm['cagr_pct']:+.1f}%)")
    print(f"  Sharpe {bestm['sharpe']:.2f} vs baseline {baseline_sharpe:.2f} (lift {lift:+.2f})")
    print(f"  DD {bestm['max_dd_pct']:.1f}% vs baseline {bm['max_dd_pct']:.1f}%")
    print(f"  Calmar {bestm['calmar']:.2f} vs baseline {bm['calmar']:.2f}")

    # ── Success gate ────────────────────────────────────────────────
    if lift >= 0.5:
        verdict = "PROMOTE"
        verdict_note = f"Sharpe lift {lift:+.2f} ≥ +0.5 target"
    elif lift >= 0.2:
        verdict = "MARGINAL"
        verdict_note = f"Sharpe lift {lift:+.2f} in [+0.2, +0.5) — does NOT hit target"
    else:
        verdict = "KILL"
        verdict_note = f"Sharpe lift {lift:+.2f} < +0.2 — below kill threshold"
    print(f"\n[verdict] {verdict}: {verdict_note}")

    # ── JSON output ─────────────────────────────────────────────────
    summary = {
        "experiment": "EXP-1950",
        "title": "Adaptive Kelly Position Sizing for EXP-1220",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "exp1220": "scripts.ultimate_portfolio.load_exp1220_dynamic (real Yahoo SPY+^VIX+^VIX3M)",
            "regime": "compass.regime.RegimeClassifier on real Yahoo SPY+^VIX",
            "sharpe_formula": "compass.metrics.full_metrics (mean/std × √252)",
        },
        "data_window": {
            "start": str(exp1220.index.min().date()),
            "end": str(exp1220.index.max().date()),
            "n_days": int(len(exp1220)),
        },
        "regime_distribution": {k: round(v, 4) for k, v in dist.items()},
        "hyperparameters": {
            "kelly_fractions": KELLY_FRACTIONS,
            "lookbacks": LOOKBACKS,
            "regime_multipliers": REGIME_MULT,
            "max_leverage": MAX_LEVERAGE,
            "baseline_leverage": BASELINE_LEV,
        },
        "baseline": {
            "name": baseline["name"],
            "leverage": baseline["leverage"],
            "metrics": bm,
            "yearly": yearly(baseline["daily_returns"]),
        },
        "variants": [
            {
                "name": v["name"],
                "lookback": v["lookback"],
                "kelly_fraction": v["kelly_fraction"],
                "regime_applied": v["regime_applied"],
                "avg_leverage": round(v["avg_leverage"], 3),
                "max_leverage": round(v["max_leverage"], 3),
                "min_leverage": round(v["min_leverage"], 3),
                "metrics": v["metrics"],
                "sharpe_lift": round(v["metrics"]["sharpe"] - baseline_sharpe, 3),
            }
            for v in variants
        ],
        "best": {
            "name": best["name"],
            "lookback": best["lookback"],
            "kelly_fraction": best["kelly_fraction"],
            "regime_applied": best["regime_applied"],
            "avg_leverage": round(best["avg_leverage"], 3),
            "metrics": bestm,
            "yearly": yearly(best["daily_returns"]),
            "sharpe_lift": round(lift, 3),
            "cagr_delta_pct": round(bestm["cagr_pct"] - bm["cagr_pct"], 2),
            "dd_delta_pct": round(bestm["max_dd_pct"] - bm["max_dd_pct"], 2),
            "calmar_delta": round(bestm["calmar"] - bm["calmar"], 2),
        },
        "success_criterion": {
            "target_sharpe_lift": 0.5,
            "kill_threshold": 0.2,
            "measured_lift": round(lift, 3),
            "verdict": verdict,
            "note": verdict_note,
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    # ── HTML ────────────────────────────────────────────────────────
    variant_rows = ""
    for v in sorted(variants, key=lambda v: -v["metrics"]["sharpe"]):
        m = v["metrics"]
        l = m["sharpe"] - baseline_sharpe
        color = "#16a34a" if l > 0.2 else ("#dc2626" if l < 0 else "#0f172a")
        variant_rows += (
            f"<tr><td>{v['name']}</td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{m['sharpe']:.2f}</td>"
            f"<td style='color:{color};font-weight:700'>{l:+.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['calmar']:.2f}</td>"
            f"<td>{v['avg_leverage']:.2f}</td></tr>"
        )

    verdict_color = (
        "#16a34a" if verdict == "PROMOTE"
        else ("#f59e0b" if verdict == "MARGINAL" else "#dc2626")
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-1950 — Adaptive Kelly Sizing</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.75em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.verdict {{ background:#fff;border:2px solid {verdict_color};border-radius:10px;padding:18px;margin:20px 0; }}
.verdict h3 {{ margin-top:0;color:{verdict_color}; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.74em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>
<h1>EXP-1950 — Adaptive Kelly Position Sizing</h1>
<p style="color:#64748b">EXP-1220 dynamic stream · rolling Kelly × regime multiplier ·
2020-2025 real Yahoo · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — real data only:</strong><br>
exp1220: scripts.ultimate_portfolio.load_exp1220_dynamic<br>
regime: compass.regime.RegimeClassifier on Yahoo SPY+^VIX<br>
Sharpe: compass.metrics.full_metrics (mean/std × √252, canonical)<br>
Rolling estimator shifted by 1 day → no look-ahead
</div>

<div class="verdict">
<h3>{verdict}: {verdict_note}</h3>
Best: <code>{best['name']}</code><br>
CAGR {bestm['cagr_pct']:+.1f}% vs {bm['cagr_pct']:+.1f}% baseline<br>
Sharpe {bestm['sharpe']:.2f} vs {baseline_sharpe:.2f} baseline (lift {lift:+.2f})<br>
Max DD {bestm['max_dd_pct']:.1f}% vs {bm['max_dd_pct']:.1f}% baseline
</div>

<h2>Variants sorted by Sharpe</h2>
<table>
<thead><tr><th>Variant</th><th>CAGR</th><th>Sharpe</th><th>Δ Sharpe</th><th>Max DD</th><th>Calmar</th><th>Avg Lev</th></tr></thead>
<tbody>{variant_rows}</tbody>
</table>

<h2>Baseline reference</h2>
<p><code>static_2.0x</code>: CAGR {bm['cagr_pct']:+.1f}% · Sharpe {bm['sharpe']:.2f} ·
DD {bm['max_dd_pct']:.1f}% · Calmar {bm['calmar']:.2f} · Vol {bm['vol_pct']:.1f}%</p>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp1950_adaptive_kelly.py · Rule Zero · real Yahoo only
</p>
</body></html>"""
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
