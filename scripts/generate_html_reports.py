import glob
import json
import os
from datetime import datetime

SNAPSHOT_DIR = "/Users/charlesbot/projects/pilotai-credit-spreads/output/historical_snapshots"
OUTPUT_DIR = "/Users/charlesbot/projects/pilotai-credit-spreads/output/html_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fff; color: #1a1a1a; max-width: 860px; margin: 0 auto; padding: 30px 20px; line-height: 1.6; }
h1 { font-size: 24px; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
h2 { color: #1e40af; margin-top: 30px; font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
.score-box { display: inline-block; padding: 8px 16px; border-radius: 8px; font-size: 28px; font-weight: 700; margin: 8px 4px; }
.score-green { background: #dcfce7; color: #166534; }
.score-yellow { background: #fef9c3; color: #854d0e; }
.score-red { background: #fee2e2; color: #991b1b; }
.score-blue { background: #dbeafe; color: #1e40af; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th { background: #1e293b; color: white; padding: 8px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) { background: #f8fafc; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.b-lead { background: #dcfce7; color: #166534; }
.b-weak { background: #fef9c3; color: #854d0e; }
.b-lag { background: #fee2e2; color: #991b1b; }
.b-imp { background: #dbeafe; color: #1e40af; }
.dim-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.dim-card { flex: 1; min-width: 140px; background: #f8fafc; border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #e5e7eb; }
.dim-label { font-size: 12px; color: #6b7280; }
.dim-val { font-size: 22px; font-weight: 700; }
"""

QUAD_BADGE = {"Leading": "b-lead", "Weakening": "b-weak", "Lagging": "b-lag", "Improving": "b-imp"}

def score_class(val):
    if val >= 65:
        return "score-green"
    if val >= 45:
        return "score-yellow"
    return "score-red"

def dim_color(val):
    if val >= 65:
        return "#16a34a"
    if val >= 45:
        return "#854d0e"
    return "#dc2626"

def generate_report(snap):
    d = snap["date"]
    dt = datetime.strptime(d, "%Y-%m-%d")
    spy = snap.get("spy_close", "N/A")
    ms = snap.get("macro_score", {})
    overall = ms.get("overall", 0)
    ind = ms.get("indicators", {})
    sectors = snap.get("sector_rankings", [])
    leading = snap.get("leading_sectors", [])
    lagging = snap.get("lagging_sectors", [])

    regime = "BULLISH" if overall >= 65 else "NEUTRAL" if overall >= 45 else "BEARISH"

    # Sector table rows
    sector_rows = ""
    for i, s in enumerate(sectors):
        q = s.get("rrg_quadrant", "?")
        bc = QUAD_BADGE.get(q, "b-imp")
        rs3 = s.get("rs_3m")
        rs3_str = f"{rs3:+.1f}%" if rs3 is not None else "N/A"
        rs3_color = "#16a34a" if rs3 and rs3 > 0 else "#dc2626" if rs3 and rs3 < 0 else "#6b7280"
        sector_rows += f"""<tr>
            <td><strong>#{i+1}</strong></td>
            <td><strong>{s.get('ticker','')}</strong></td>
            <td>{s.get('name','')}</td>
            <td style="color:{rs3_color};font-weight:600">{rs3_str}</td>
            <td><span class="badge {bc}">{q}</span></td>
        </tr>"""

    # Macro dimensions
    dims_html = ""
    for key, label in [("growth","Growth"),("inflation","Inflation"),("fed_policy","Fed Policy"),("risk_appetite","Risk Appetite")]:
        val = ms.get(key, 0)
        dims_html += f"""<div class="dim-card">
            <div class="dim-label">{label}</div>
            <div class="dim-val" style="color:{dim_color(val)}">{val:.0f}</div>
        </div>"""

    # Indicators
    ind_rows = ""
    ind_labels = {
        "vix": ("VIX", ""), "t10y2y": ("10Y-2Y Spread", "%"), "fedfunds": ("Fed Funds Rate", "%"),
        "cpi_yoy_pct": ("CPI YoY", "%"), "core_cpi_yoy_pct": ("Core CPI YoY", "%"),
        "breakeven_5y": ("5Y Breakeven", "%"), "hy_oas_pct": ("HY Credit Spread", "%"),
        "cfnai_3m": ("CFNAI 3M Avg", ""), "payrolls_3m_avg_k": ("Payrolls 3M Avg", "K")
    }
    for k, (label, suffix) in ind_labels.items():
        v = ind.get(k)
        if v is not None:
            ind_rows += f"<tr><td>{label}</td><td><strong>{v:.2f}{suffix}</strong></td></tr>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Macro Snapshot — {d}</title><style>{CSS}</style></head><body>
<h1>📊 Weekly Macro Snapshot — {dt.strftime('%B %d, %Y')}</h1>
<p style="color:#6b7280">SPY Close: <strong>${spy}</strong> · Regime: <strong>{regime}</strong></p>

<h2>🧠 Macro Score</h2>
<div class="score-box {score_class(overall)}">{overall:.1f} / 100</div>
<span style="font-size:16px;color:#6b7280;margin-left:8px">{regime}</span>

<div class="dim-row">{dims_html}</div>

<h2>📈 Sector Rankings (by 3-Month Relative Strength vs SPY)</h2>
<table>
<tr><th>Rank</th><th>Ticker</th><th>Sector</th><th>RS vs SPY</th><th>RRG Quadrant</th></tr>
{sector_rows}
</table>

<h2>🔥 Hot Sectors (Leading)</h2>
<p style="font-size:15px">{"  ".join(f'<span class="badge b-lead">{s}</span>' for s in leading) if leading else "None"}</p>

<h2>🧊 Cold Sectors (Lagging)</h2>
<p style="font-size:15px">{"  ".join(f'<span class="badge b-lag">{s}</span>' for s in lagging) if lagging else "None"}</p>

<h2>📉 Economic Indicators</h2>
<table><tr><th>Indicator</th><th>Value</th></tr>{ind_rows}</table>

<div style="margin-top:30px;padding:16px;background:#f8fafc;border-radius:8px;text-align:center;color:#6b7280;font-size:13px">
PilotAI Macro Intelligence · Week of {d}
</div>
</body></html>"""
    return html

# Generate all
count = 0
for year_dir in sorted(glob.glob(f"{SNAPSHOT_DIR}/20*")):
    year = os.path.basename(year_dir)
    os.makedirs(f"{OUTPUT_DIR}/{year}", exist_ok=True)
    for jf in sorted(glob.glob(f"{year_dir}/*.json")):
        with open(jf) as f:
            snap = json.load(f)
        html = generate_report(snap)
        fname = os.path.basename(jf).replace(".json", ".html")
        with open(f"{OUTPUT_DIR}/{year}/{fname}", "w") as f:
            f.write(html)
        count += 1

# Also generate an index page
index_rows = ""
for year_dir in sorted(glob.glob(f"{OUTPUT_DIR}/20*")):
    year = os.path.basename(year_dir)
    files = sorted(glob.glob(f"{year_dir}/*.html"))
    index_rows += f"<h3>{year} ({len(files)} reports)</h3><ul>"
    for fp in files:
        fname = os.path.basename(fp)
        date = fname.replace(".html","")
        index_rows += f'<li><a href="{year}/{fname}">{date}</a></li>'
    index_rows += "</ul>"

index_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Macro Snapshots Index</title><style>{CSS}</style></head><body>
<h1>📊 Macro Snapshot Archive (2020–2026)</h1>
<p>{count} weekly reports</p>{index_rows}</body></html>"""
with open(f"{OUTPUT_DIR}/index.html", "w") as f:
    f.write(index_html)

print(f"Generated {count} HTML reports + index.html")
