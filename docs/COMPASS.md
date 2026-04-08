# 🧭 COMPASS — Composite Macro Position & Sector Signal

## What is COMPASS?

COMPASS is PilotAI's proprietary macro intelligence indicator that scores the market environment weekly across four dimensions, ranks sectors by relative strength, and provides actionable trade direction signals.

**One sentence:** COMPASS tells the trading system *where* to trade, *how big* to trade, and *when to back off* — based on macroeconomics, not just chart patterns.

## The Name

**C**omposite **O**verall **M**acro **P**osition **A**nd **S**ector **S**ignal

## Core Components

### 1. Macro Score (0–100)
Four dimensions, equally weighted:

| Dimension | What It Measures | Key Data Sources |
|-----------|-----------------|------------------|
| **Growth** | Economic expansion/contraction | CFNAI, Nonfarm Payrolls |
| **Inflation** | Price stability (Goldilocks = 2–2.5%) | CPI, Core CPI, 5Y Breakeven |
| **Fed Policy** | Monetary policy stance | 10Y-2Y spread, Fed Funds rate |
| **Risk Appetite** | Market fear/greed | VIX, HY credit spread |

**Regimes:**
- 65–100: BULLISH 📈
- 45–64: NEUTRAL ➡️
- 0–44: BEARISH 📉

### 2. Sector Relative Strength (RS)
- Tracks 15 ETFs (11 SPDR sectors + SOXX, XBI, PAVE, ITA)
- 3-month and 12-month RS vs SPY
- Ranks sectors weekly

### 3. RRG Quadrants (Relative Rotation Graphs)
Each sector is classified into one of four quadrants:
- **Leading** 🟢 — Outperforming AND accelerating → Bull put spreads
- **Improving** 🔵 — Recovering → Watch / small positions
- **Weakening** 🟡 — Decelerating → Iron condors / reduce
- **Lagging** 🔴 — Underperforming → Bear call spreads

### 4. Event Scaling
Reduces position size before major macro events:
- FOMC: ramps from 1.0x → 0.5x over 5 days
- CPI: ramps from 1.0x → 0.65x over 2 days
- NFP: ramps from 1.0x → 0.75x over 2 days

## Key Finding

COMPASS score has a **negative correlation (-0.13)** with forward returns:
- Low score (fear) → avg **+4.73%** over 12 weeks
- High score (calm) → avg **+2.27%** over 12 weeks

**Translation:** Buy when COMPASS says the market is scared. Trim when it says everyone's comfortable.

## Data Coverage

- **Historical:** 323 weekly snapshots (January 2020 – March 2026)
- **Data sources:** FRED (macro indicators), Polygon.io (sector ETF prices)
- **Update frequency:** Weekly (Fridays 5PM ET) + daily event gate (6AM ET)
- **Lookahead bias protection:** RELEASE_LAG_DAYS prevents using data not yet published on historical dates

## Where COMPASS Lives

| Component | Location |
|-----------|----------|
| Engine | `shared/macro_snapshot_engine.py` |
| DB layer | `shared/macro_state_db.py` |
| Event gate | `shared/macro_event_gate.py` |
| Database | `data/macro_state.db` |
| CLI | `scripts/run_macro_snapshot.py` |
| Report gen | `scripts/macro_report.py` |
| API server | `scripts/macro_api.py` |
| Historical snapshots | `output/historical_snapshots/` |

## API Access

**Live API:** https://pilotai-macro-intelligence-production.up.railway.app
**Swagger docs:** https://pilotai-macro-intelligence-production.up.railway.app/docs
**Auth:** `X-API-Key: dev-pilotai-macro-2026`

### Python Integration
```python
from shared.macro_state_db import (
    get_current_macro_score,      # → 60.4
    get_sector_rankings,          # → [{"ticker": "XLE", "rs_3m": 24.9, ...}]
    get_event_scaling_factor,     # → 1.00 (or 0.50 near FOMC)
    get_eligible_underlyings,     # → ["SPY", "QQQ", "IWM", "XLE"]
)
```

## Experiment Series

- **exp_090:** Champion baseline (MA200 filter, no COMPASS)
- **exp_100+:** COMPASS-integrated experiments (proving macro alpha)

---

*COMPASS was conceived by Carlos Cruz on March 7, 2026 and built from idea to production in 5 hours.* 🧭
