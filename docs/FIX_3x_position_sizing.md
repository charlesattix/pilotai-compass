# FIX: EXP-1220 3x Paper Trading — Position Sizing Bug

**Date:** 2026-04-06
**Reported by:** Carlos (only 3 contracts being placed instead of expected 15-20)
**Diagnosed by:** Maximus

---

## Problem

The 3x paper trading account is only placing **3 contracts per trade** (~$1,500 max risk) instead of the expected **15-20 contracts** (~$10,000-$15,000 max risk) for a 3x leverage configuration on a $100k account.

## Root Cause

**Config field name mismatch between YAML and code.**

The position sizer (`alerts/alert_position_sizer.py`) reads from the `risk:` section:
```python
# Line 297 — what the code looks for:
raw_risk_pct = float(risk_cfg.get("max_risk_per_trade", 5.0)) / 100.0

# Line 330 — what the code looks for:
max_contracts = int(risk_cfg.get("max_contracts", 25))
```

But `configs/paper_exp1220_3x.yaml` has position sizing under `sizing:` with different field names:
```yaml
# What the YAML has (NEVER READ by the sizer):
sizing:
  base_risk_pct: 1.0
  leveraged_risk_pct: 3.0
  max_portfolio_risk_pct: 14.0
  contracts_min: 1
  contracts_max: 20
```

**Result:** The sizer never finds `max_risk_per_trade` in the `risk:` section, so it defaults to **5%** — but since the VIX is elevated and macro scaling may reduce this, plus the flat sizing base is $100k (not leveraged), the final contract count comes out very low.

Additionally, `starting_capital` is looked up under `backtest:` (not `account:`), so it also defaults.

## The Fix

### Option A: Fix the Config (Quick — recommended)

Add the correct field names to the `risk:` section in `configs/paper_exp1220_3x.yaml`:

```yaml
risk:
  max_risk_per_trade: 5.0      # 5% of starting capital per trade
  max_contracts: 20             # hard cap per trade
  min_contracts: 1
  sizing_mode: flat             # use starting_capital as base
  max_daily_loss_pct: 5.0
  max_weekly_loss_pct: 9.0
  max_drawdown_halt_pct: 16.0
  drawdown_recovery_pct: 8.0
  correlation_check: true
```

And add `starting_capital` under `backtest:`:
```yaml
backtest:
  starting_capital: 100000
```

**Expected result with this fix:**
- $100k × 5% = $5,000 risk per trade
- $5 spread - $0.56 credit = $4.44 max loss = $444/contract
- $5,000 ÷ $444 = **11 contracts** per trade
- With VIX/macro scaling, probably 8-11 contracts

### Option B: Fix the Code (Better long-term)

Update `_flat_risk_size()` in `alerts/alert_position_sizer.py` to also check the `sizing:` section as a fallback:

```python
# Around line 280, add sizing config fallback:
risk_cfg = self.config.get("risk", {})
sizing_cfg = self.config.get("sizing", {})
strategy_cfg = self.config.get("strategy", {})
backtest_cfg = self.config.get("backtest", {})
account_cfg = self.config.get("account", {})

# Merge sizing into risk as fallback (risk takes priority)
effective_risk = {**sizing_cfg, **risk_cfg}

# Then use effective_risk instead of risk_cfg throughout
```

And add field name aliases:
```python
raw_risk_pct = float(effective_risk.get(
    "max_risk_per_trade",
    effective_risk.get("leveraged_risk_pct",
    effective_risk.get("base_risk_pct", 5.0))
)) / 100.0

max_contracts = int(effective_risk.get(
    "max_contracts",
    effective_risk.get("contracts_max", 25)
))
```

### Option C: Both (Recommended)

Fix the config now for immediate effect, then fix the code to prevent this class of bug in all configs.

## How to Apply (Mac Studio)

1. Edit the config:
```bash
cd /path/to/pilotai-credit-spreads
nano configs/paper_exp1220_3x.yaml
# Add max_risk_per_trade and max_contracts to the risk: section (see Option A above)
```

2. Restart the scanner LaunchAgent:
```bash
launchctl kickstart -k gui/$(id -u)/com.pilotai.exp1220-3x
# Or whatever the LaunchAgent is named
```

3. Verify on next trade that contract count is 8-11+ instead of 3.

## Also Affected

Check these configs for the same issue:
- `configs/paper_exp1220_2x.yaml`
- `configs/paper_exp1220_4x.yaml`
- `configs/paper_exp1220_5x.yaml`

They likely have the same `sizing:` vs `risk:` mismatch.

---

**Priority:** HIGH — Every trade at 3 contracts is earning ~5x less than it should.
