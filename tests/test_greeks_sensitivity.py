"""Tests for compass/greeks_sensitivity.py — Greeks scenario analysis."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.greeks_sensitivity import (
    GreekProfile,
    GreeksSensitivityAnalyzer,
    GreeksSnapshot,
    OptimalEntry,
    ScenarioCell,
    SensitivitySummary,
    bs_call_price,
    bs_put_price,
    build_scenario_matrix,
    call_spread_value,
    compute_greeks,
    put_spread_value,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"


def _make_trades(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "entry_date": pd.bdate_range("2023-01-02", periods=n),
        "spy_price": rng.uniform(430, 470, n),
        "short_strike": rng.uniform(420, 450, n),
        "spread_width": np.full(n, 5.0),
        "net_credit": rng.uniform(0.30, 1.20, n),
        "dte_at_entry": rng.randint(10, 45, n),
        "vix": rng.uniform(14, 40, n),
        "iv_rank": rng.uniform(10, 90, n),
        "otm_pct": rng.uniform(1, 6, n),
        "regime": rng.choice(["bull", "bear", "high_vol", "low_vol"], n),
        "pnl": rng.normal(20, 200, n),
        "win": rng.choice([0, 1], n, p=[0.35, 0.65]),
        "spread_type": "bull_put",
    })


# ── Black-Scholes primitives ─────────────────────────────────────────────


class TestBSPricing:

    def test_put_price_positive(self):
        p = bs_put_price(S=450, K=440, T=21/252, sigma=0.20)
        assert p > 0

    def test_put_price_deep_itm(self):
        p = bs_put_price(S=400, K=450, T=21/252, sigma=0.20)
        assert p > 45  # near intrinsic

    def test_put_price_at_expiry(self):
        p = bs_put_price(S=450, K=440, T=0, sigma=0.20)
        assert p == 0.0  # OTM at expiry

    def test_call_price_positive(self):
        c = bs_call_price(S=450, K=460, T=21/252, sigma=0.20)
        assert c > 0

    def test_put_call_parity_approx(self):
        S, K, T, sigma, r = 450, 445, 30/252, 0.20, 0.045
        c = bs_call_price(S, K, T, sigma, r)
        p = bs_put_price(S, K, T, sigma, r)
        parity = c - p - (S - K * math.exp(-r * T))
        assert abs(parity) < 0.05

    def test_put_spread_positive_when_otm(self):
        v = put_spread_value(S=460, K_short=450, K_long=445, T=21/252, sigma=0.20)
        assert v >= 0

    def test_call_spread_positive_when_otm(self):
        v = call_spread_value(S=440, K_short=450, K_long=455, T=21/252, sigma=0.20)
        assert v >= 0


# ── compute_greeks ───────────────────────────────────────────────────────


class TestComputeGreeks:

    def test_returns_snapshot(self):
        g = compute_greeks(S=450, K_short=440, K_long=435, T=21/252, sigma=0.20)
        assert isinstance(g, GreeksSnapshot)

    def test_delta_negative_for_put_spread(self):
        g = compute_greeks(S=450, K_short=445, K_long=440, T=21/252, sigma=0.20)
        # Bull put spread: net short delta (short higher-strike put)
        assert g.delta < 0

    def test_theta_positive_for_credit_spread(self):
        g = compute_greeks(S=460, K_short=445, K_long=440, T=30/252, sigma=0.20)
        # OTM credit spread: time decay benefits seller (theta > 0)
        assert g.theta > 0

    def test_vega_positive_for_spread_value(self):
        g = compute_greeks(S=460, K_short=445, K_long=440, T=30/252, sigma=0.20)
        # Spread value (cost to close) rises with vol → vega > 0
        # For the SELLER, P&L vega = -spread vega (rising vol hurts)
        assert g.vega > 0

    def test_greeks_finite(self):
        g = compute_greeks(S=450, K_short=440, K_long=435, T=21/252, sigma=0.20)
        assert all(math.isfinite(v) for v in [g.delta, g.gamma, g.theta, g.vega, g.rho])


# ── build_scenario_matrix ────────────────────────────────────────────────


class TestScenarioMatrix:

    def test_returns_list(self):
        cells = build_scenario_matrix(
            S=450, K_short=445, K_long=440, sigma=0.20, credit=0.65,
        )
        assert isinstance(cells, list)
        assert len(cells) > 100

    def test_cell_has_fields(self):
        cells = build_scenario_matrix(
            S=450, K_short=445, K_long=440, sigma=0.20, credit=0.65,
        )
        c = cells[0]
        assert hasattr(c, "price_pct")
        assert hasattr(c, "iv_pct")
        assert hasattr(c, "dte")
        assert hasattr(c, "pnl_pct")

    def test_at_entry_pnl_near_zero(self):
        cells = build_scenario_matrix(
            S=450, K_short=445, K_long=440, sigma=0.20, credit=0.65,
            dte_max=45,
        )
        # At price=0%, IV=0%, DTE=45 (entry conditions) → PnL ≈ 0
        entry_cells = [c for c in cells if c.price_pct == 0 and c.iv_pct == 0 and c.dte == 45]
        assert len(entry_cells) >= 1
        assert abs(entry_cells[0].pnl_pct) < 10

    def test_expiry_otm_is_max_profit(self):
        cells = build_scenario_matrix(
            S=460, K_short=445, K_long=440, sigma=0.20, credit=0.65,
            dte_max=45,
        )
        # OTM at expiry with no price move → near max profit
        expiry = [c for c in cells if c.price_pct == 0 and c.iv_pct == 0 and c.dte == 0]
        assert len(expiry) >= 1
        assert expiry[0].pnl_pct > 50

    def test_custom_ranges(self):
        cells = build_scenario_matrix(
            S=450, K_short=445, K_long=440, sigma=0.20, credit=0.65,
            price_range_pct=2.0, price_step_pct=1.0,
            iv_range_pct=10.0, iv_step_pct=5.0,
            dte_max=20, dte_step=10,
        )
        # 5 price × 5 IV × 3 DTE = 75 cells
        assert 50 < len(cells) < 100


# ── GreeksSensitivityAnalyzer ────────────────────────────────────────────


class TestAnalyzer:

    def test_fit_returns_self(self):
        a = GreeksSensitivityAnalyzer()
        assert a.fit(_make_trades()) is a

    def test_summary_populated(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        s = a.summary()
        assert isinstance(s, SensitivitySummary)
        assert s.n_trades == 60

    def test_scenario_matrix_populated(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        assert len(a.summary().scenario_matrix) > 100

    def test_greeks_at_entry(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        g = a.summary().greeks_at_entry
        assert g is not None
        assert g.delta != 0

    def test_regime_profiles(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        profiles = a.summary().regime_profiles
        assert len(profiles) >= 2
        regimes = {p.regime for p in profiles}
        assert "bull" in regimes

    def test_optimal_entries(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        opts = a.summary().optimal_entries
        assert len(opts) >= 1
        assert opts[0].n_trades > 0
        assert 0 <= opts[0].win_rate <= 100

    def test_empty_df(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(pd.DataFrame())
        assert a.summary().n_trades == 0

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data(self):
        df = pd.read_csv(EXP400_CSV)
        a = GreeksSensitivityAnalyzer()
        a.fit(df)
        s = a.summary()
        assert s.n_trades > 200
        assert len(s.regime_profiles) >= 2


# ── HTML report ──────────────────────────────────────────────────────────


class TestReport:

    def test_returns_html(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        html = a.generate_report()
        assert "<!DOCTYPE html>" in html

    def test_contains_sections(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        html = a.generate_report()
        assert "Scenario Heatmap" in html
        assert "Optimal Entry" in html
        assert "Regime" in html

    def test_writes_to_file(self):
        a = GreeksSensitivityAnalyzer()
        a.fit(_make_trades())
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "r.html")
            a.generate_report(p)
            assert Path(p).exists()

    def test_unfitted(self):
        html = GreeksSensitivityAnalyzer().generate_report()
        assert "No data" in html
