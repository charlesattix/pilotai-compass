"""Tests for compass.anomaly_detector – anomaly detection system."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.anomaly_detector import (
    CRITICAL,
    INFO,
    WARNING,
    AlertMessage,
    Anomaly,
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    AnomalyStats,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_market_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Normal market data with one injected spike."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 450.0 * np.cumprod(1 + rng.randn(n) * 0.008)
    volume = 1e6 + rng.randn(n) * 1e5
    iv = 18.0 + rng.randn(n) * 2.0
    # Inject anomalies
    prices[150] = prices[149] * 0.90  # 10% gap
    volume[120] = 5e6                 # volume spike
    iv[130] = 45.0                    # IV spike
    return pd.DataFrame({
        "date": idx,
        "close": prices,
        "volume": volume,
        "iv": iv,
    })


def _make_trades(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    pnl = rng.randn(n) * 50 + 5
    slippage = np.abs(rng.randn(n) * 0.01)
    fill_time = np.abs(rng.randn(n) * 2 + 5)
    # Inject anomalies
    pnl[80] = -500                    # outlier loss
    slippage[90] = 0.50               # slippage spike
    fill_time[100] = 60.0             # slow fill
    return pd.DataFrame({
        "date": idx,
        "pnl": pnl,
        "slippage": slippage,
        "fill_time": fill_time,
    })


def _make_clean_market(n: int = 100, seed: int = 99) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": idx,
        "close": 450.0 + np.cumsum(rng.randn(n) * 0.5),
        "volume": 1e6 + rng.randn(n) * 5e4,
    })


# ── Constructor ─────────────────────────────────────────────────────────────
class TestAnomalyDetectorInit:
    def test_defaults(self):
        d = AnomalyDetector()
        assert d.config.zscore_warn == 2.0
        assert d.config.zscore_crit == 3.0

    def test_custom_config(self):
        cfg = AnomalyConfig(zscore_warn=1.5, zscore_crit=2.5, lookback=30)
        d = AnomalyDetector(config=cfg)
        assert d.config.zscore_warn == 1.5
        assert d.config.lookback == 30


# ── Market detection ────────────────────────────────────────────────────────
class TestMarketAnomalies:
    def test_detects_market_anomalies(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        market = [a for a in result.anomalies if a.category == "market"]
        assert len(market) > 0

    def test_price_gap_detected(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        gaps = [a for a in result.anomalies if a.anomaly_type == "price_gap"]
        assert len(gaps) > 0

    def test_volume_spike_detected(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        vol = [a for a in result.anomalies if a.anomaly_type == "unusual_volume"]
        assert len(vol) > 0

    def test_iv_spike_detected(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        iv = [a for a in result.anomalies if a.anomaly_type == "iv_spike"]
        assert len(iv) > 0

    def test_clean_data_few_anomalies(self):
        result = AnomalyDetector().detect(market_data=_make_clean_market())
        # Clean data should produce far fewer anomalies than spiked data
        spiked = AnomalyDetector().detect(market_data=_make_market_data())
        assert len(result.anomalies) <= len(spiked.anomalies)


# ── Trade detection ─────────────────────────────────────────────────────────
class TestTradeAnomalies:
    def test_detects_trade_anomalies(self):
        result = AnomalyDetector().detect(trades=_make_trades())
        trade = [a for a in result.anomalies if a.category == "trade"]
        assert len(trade) > 0

    def test_outlier_pnl_detected(self):
        result = AnomalyDetector().detect(trades=_make_trades())
        pnl = [a for a in result.anomalies if a.anomaly_type == "outlier_pnl"]
        assert len(pnl) > 0

    def test_slippage_spike_detected(self):
        result = AnomalyDetector().detect(trades=_make_trades())
        slip = [a for a in result.anomalies if a.anomaly_type == "abnormal_slippage"]
        assert len(slip) > 0

    def test_unusual_fill_time_detected(self):
        result = AnomalyDetector().detect(trades=_make_trades())
        ft = [a for a in result.anomalies if a.anomaly_type == "unusual_fill_time"]
        assert len(ft) > 0


# ── Severity ────────────────────────────────────────────────────────────────
class TestSeverity:
    def test_severity_values_valid(self):
        result = AnomalyDetector().detect(
            market_data=_make_market_data(), trades=_make_trades(),
        )
        for a in result.anomalies:
            assert a.severity in (INFO, WARNING, CRITICAL)

    def test_critical_has_high_zscore(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        crits = [a for a in result.anomalies if a.severity == CRITICAL]
        for c in crits:
            assert abs(c.zscore) >= 2.0  # at least warning-level z

    def test_sensitivity_increases_with_lower_thresholds(self):
        strict = AnomalyDetector(config=AnomalyConfig(zscore_warn=1.0, zscore_crit=2.0))
        loose = AnomalyDetector(config=AnomalyConfig(zscore_warn=3.0, zscore_crit=4.0))
        r_strict = strict.detect(market_data=_make_market_data())
        r_loose = loose.detect(market_data=_make_market_data())
        assert len(r_strict.anomalies) >= len(r_loose.anomalies)


# ── Alerts ──────────────────────────────────────────────────────────────────
class TestAlerts:
    def test_alerts_for_warn_and_crit(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        assert len(result.alerts) > 0
        for alert in result.alerts:
            assert alert.severity in (WARNING, CRITICAL)

    def test_alert_fields(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        if result.alerts:
            a = result.alerts[0]
            assert len(a.title) > 0
            assert len(a.body) > 0
            assert len(a.timestamp) > 0

    def test_telegram_format(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        if result.alerts:
            msg = AnomalyDetector.format_telegram(result.alerts[0])
            assert isinstance(msg, str)
            assert "*" in msg  # markdown bold
            assert "```" in msg  # code block

    def test_no_alerts_for_info_only(self):
        # Very loose thresholds → only info severity → no alerts
        loose = AnomalyDetector(config=AnomalyConfig(zscore_warn=50.0, zscore_crit=100.0, iqr_multiplier=100.0))
        result = loose.detect(market_data=_make_clean_market())
        assert len(result.alerts) == 0


# ── History ─────────────────────────────────────────────────────────────────
class TestHistory:
    def test_history_accumulates(self):
        d = AnomalyDetector()
        d.detect(market_data=_make_market_data())
        d.detect(trades=_make_trades())
        assert len(d.get_history()) > 0

    def test_clear_history(self):
        d = AnomalyDetector()
        d.detect(market_data=_make_market_data())
        d.clear_history()
        assert len(d.get_history()) == 0


# ── Stats ───────────────────────────────────────────────────────────────────
class TestStats:
    def test_stats_present(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        assert result.stats is not None
        assert result.stats.total == len(result.anomalies)

    def test_severity_counts_sum(self):
        result = AnomalyDetector().detect(
            market_data=_make_market_data(), trades=_make_trades(),
        )
        s = result.stats
        assert s.info_count + s.warning_count + s.critical_count == s.total

    def test_by_type_populated(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        assert len(result.stats.by_type) > 0

    def test_by_category_populated(self):
        result = AnomalyDetector().detect(
            market_data=_make_market_data(), trades=_make_trades(),
        )
        assert "market" in result.stats.by_category
        assert "trade" in result.stats.by_category


# ── Edge cases ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_no_data(self):
        result = AnomalyDetector().detect()
        assert result.anomalies == []

    def test_empty_dataframes(self):
        result = AnomalyDetector().detect(
            market_data=pd.DataFrame(), trades=pd.DataFrame(),
        )
        assert result.anomalies == []

    def test_generated_at_set(self):
        result = AnomalyDetector().detect(market_data=_make_market_data())
        assert len(result.generated_at) > 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = AnomalyDetector()
            result = d.detect(market_data=_make_market_data(), trades=_make_trades())
            path = d.generate_report(result, output_path=Path(tmp) / "ad.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = AnomalyDetector()
            result = d.detect(market_data=_make_market_data(), trades=_make_trades())
            path = d.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Anomaly Detection" in html
            assert "Severity" in html
            assert "Anomaly Type" in html
            assert "Timeline" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = AnomalyDetector()
            result = d.detect(market_data=_make_market_data())
            path = d.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_anomaly(self):
        a = Anomaly("2024-01-01", "market", "price_gap", WARNING, -0.05, 0.02, -3.5)
        assert a.severity == WARNING

    def test_alert_message(self):
        m = AlertMessage(CRITICAL, "Test", "body", "2024-01-01")
        assert m.severity == CRITICAL

    def test_anomaly_stats_defaults(self):
        s = AnomalyStats()
        assert s.total == 0

    def test_anomaly_result_defaults(self):
        r = AnomalyResult()
        assert r.anomalies == []
        assert r.alerts == []
