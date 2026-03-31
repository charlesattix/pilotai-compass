# Known Test Failures

**Last updated:** 2026-03-20
**Branch:** compass-v2

---

## test_hardening.py — 3 failures in TestExecutionEngineMarketGuard

### Affected Tests

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| `test_market_closed_blocks_submission` | `status == "market_closed"` | `status == "insufficient_buying_power"` | FAIL |
| `test_market_open_allows_submission` | `status == "submitted"` | `status == "insufficient_buying_power"` | FAIL |
| `test_market_closed_still_writes_db_record` | 1 trade in DB | 0 trades in DB | FAIL |

### Root Cause

A **buying power pre-check** ("Lesson 005") was added to `execution_engine.py:submit_opportunity()` at line 302. This check calls `self.alpaca.get_account()` to read `options_buying_power` and blocks the order if buying power is insufficient.

The execution order in `submit_opportunity()` is now:

```
1. Duplicate check
2. Drawdown circuit breaker
3. Buying power check     ← NEW (Lesson 005, line 302)
4. Position limits check
5. DB write (upsert_trade)
6. Market hours guard     ← tests expect this to fire
7. Alpaca order submission
```

The `_make_alpaca()` helper in `test_hardening.py` (line 45) does **not** mock `get_account()`. When the buying power check calls `mock_alpaca.get_account()`, MagicMock auto-creates it and returns another MagicMock. Then `float(account.get("options_buying_power", 0))` evaluates to `0.0` (the fallback `0`), and `$0.00 < $500.00` triggers the buying power block — before execution ever reaches the market hours guard or DB write.

### Why It's Not a Bug in Production

The buying power check is correct production behavior — it prevents orders when the account lacks margin. The test mock just wasn't updated to provide a realistic `get_account()` response.

### Phase 2 Blocker?

**No.** These 3 tests are in `test_hardening.py` which tests execution engine hardening (H5-H6 market hours guard). They do not touch any compass/ module, regime classifier, event gate, or sizing code. Phase 2 (compass/ package creation) and Phase 3 (CC-2 test work) are unaffected.

### Proposed Fix

Update `_make_alpaca()` in `test_hardening.py` to include a `get_account()` mock with sufficient buying power:

```python
def _make_alpaca(positions=None, order_status=None, market_clock=None):
    mock = MagicMock()
    mock.get_positions.return_value = positions or []
    mock.get_order_status.return_value = order_status or {}
    mock.close_spread.return_value = {"status": "submitted", "order_id": "ord-001"}
    mock.close_iron_condor.return_value = {"status": "submitted", "order_id": "ord-ic-001"}
    mock.get_market_clock.return_value = market_clock or {"is_open": True}
    # Lesson 005: buying power check needs realistic account data
    mock.get_account.return_value = {"options_buying_power": "1000000.00"}
    ...
```

This is a one-line fix. The 30 passing tests in `test_hardening.py` are unaffected because they either:
- Use dry-run mode (`alpaca_provider=None`), bypassing the check
- Use `_make_alpaca()` but their code paths don't reach `submit_opportunity()`
- Test `market_clock=None` which still fails at buying power before market check

**Effort:** < 5 minutes. Should be fixed by whoever next touches execution tests.
