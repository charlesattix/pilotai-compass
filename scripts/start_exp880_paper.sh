#!/usr/bin/env bash
# ============================================================================
# start_exp880_paper.sh — Launch EXP-880 Crisis Hedge V2 Paper Trading
# ============================================================================
#
# Prerequisites:
#   1. Copy .env.exp880.example to .env.exp880 and fill in credentials
#   2. Ensure data/options_cache.db exists (run scripts/iron_vault_setup.py)
#   3. Ensure ML models are trained (compass/production_ensemble.py)
#
# Usage:
#   ./scripts/start_exp880_paper.sh           # normal start
#   ./scripts/start_exp880_paper.sh --dry-run # validate config only
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Validate environment ─────────────────────────────────────────────────

ENV_FILE=".env.exp880"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "  Copy .env.exp880.example to .env.exp880 and fill in credentials."
    exit 1
fi

# Check required vars
source "$ENV_FILE"
for var in ALPACA_API_KEY ALPACA_API_SECRET POLYGON_API_KEY; do
    if [ -z "${!var:-}" ] || [ "${!var}" = "your_paper_api_key_here" ] || [ "${!var}" = "your_polygon_key_here" ]; then
        echo "ERROR: $var not set or still has placeholder value in $ENV_FILE"
        exit 1
    fi
done

# ── Validate data ────────────────────────────────────────────────────────

if [ ! -f "data/options_cache.db" ]; then
    echo "WARNING: data/options_cache.db not found. IronVault cache may be empty."
    echo "  Run: python scripts/iron_vault_setup.py"
fi

# ── Create directories ───────────────────────────────────────────────────

mkdir -p data/exp880 logs output/backtest_reports

# ── Dry run mode ─────────────────────────────────────────────────────────

if [ "${1:-}" = "--dry-run" ]; then
    echo "=== EXP-880 Dry Run ==="
    echo "  Config: configs/paper_exp880.yaml"
    echo "  Env: $ENV_FILE"
    echo "  DB: data/exp880/pilotai_exp880.db"
    echo "  Log: logs/paper_exp880.log"
    echo ""
    echo "  Validating YAML config..."
    python3 -c "
import yaml
with open('configs/paper_exp880.yaml') as f:
    cfg = yaml.safe_load(f)
print(f\"  experiment_id: {cfg['experiment_id']}\")
print(f\"  paper_mode: {cfg['paper_mode']}\")
print(f\"  tickers: {cfg['tickers']}\")
print(f\"  crisis_hedge.enabled: {cfg['crisis_hedge']['enabled']}\")
print(f\"  crisis_hedge.version: {cfg['crisis_hedge']['version']}\")
print(f\"  crisis_hedge.min_scale: {cfg['crisis_hedge']['min_scale']}\")
print(f\"  crisis_hedge.dd_start: {cfg['crisis_hedge']['dd_start']}\")
print(f\"  crisis_hedge.dd_full: {cfg['crisis_hedge']['dd_full']}\")
print(f\"  leverage.base: {cfg['strategy']['leverage']['base_leverage']}\")
print(f\"  ensemble_threshold: {cfg['strategy']['ml_enhanced']['ensemble_threshold']}\")
print(f\"  confidence_sizing: {cfg['strategy']['ml_enhanced']['confidence_sizing']}\")
print(f\"  risk.drawdown_cb_pct: {cfg['risk']['drawdown_cb_pct']}\")
print(f\"  risk.max_positions: {cfg['risk']['max_positions']}\")
print()
print('  ✓ Config valid')
" 2>&1
    echo ""
    echo "  ✓ Dry run complete. Run without --dry-run to start."
    exit 0
fi

# ── Start paper trader ───────────────────────────────────────────────────

echo "============================================================"
echo "  Starting EXP-880: Crisis Hedge V2 Paper Trading"
echo "============================================================"
echo "  Config:    configs/paper_exp880.yaml"
echo "  Leverage:  2x base with regime multipliers"
echo "  Hedge:     V2 Ultra-Safe (min_scale=0.20, DD 2%→7%)"
echo "  Signal:    3-model ensemble, P>=0.75"
echo "  DD limit:  12% hard stop"
echo "============================================================"
echo ""

exec python3 main.py scheduler \
    --config configs/paper_exp880.yaml \
    --env-file "$ENV_FILE" \
    2>&1 | tee -a logs/paper_exp880.log
