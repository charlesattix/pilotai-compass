"""Tests for compass.crypto data collectors.

All HTTP calls are mocked — no real network requests.
"""

import unittest
from unittest.mock import MagicMock, call, patch

import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(json_data):
    """Return a mock requests.Response that succeeds with json_data."""
    m = MagicMock()
    m.json.return_value = json_data
    m.raise_for_status.return_value = None
    return m


def _err(status=500):
    """Return a mock requests.Response that raises HTTPError on raise_for_status."""
    m = MagicMock()
    m.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return m


# ===========================================================================
# CoinGecko
# ===========================================================================

class TestCoinGecko(unittest.TestCase):

    def setUp(self):
        # Reset the module-level rate-limit tracker before each test so tests
        # don't interfere with each other's throttle state.
        import compass.crypto.coingecko as cg
        cg._last_call_ts = 0.0

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_price_happy(self, mock_get, _mono, _sleep):
        mock_get.return_value = _ok({"bitcoin": {"usd": 65000.0}})
        from compass.crypto.coingecko import get_btc_price
        result = get_btc_price()
        self.assertEqual(result, 65000.0)
        mock_get.assert_called_once()

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_eth_price_happy(self, mock_get, _mono, _sleep):
        mock_get.return_value = _ok({"ethereum": {"usd": 3200.5}})
        from compass.crypto.coingecko import get_eth_price
        result = get_eth_price()
        self.assertEqual(result, 3200.5)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_price_api_failure_returns_none(self, mock_get, _mono, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.coingecko import get_btc_price
        result = get_btc_price()
        self.assertIsNone(result)
        # Should have retried 3 times
        self.assertEqual(mock_get.call_count, 3)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_price_malformed_response_returns_none(self, mock_get, _mono, _sleep):
        mock_get.return_value = _ok({"wrong_key": {}})
        from compass.crypto.coingecko import get_btc_price
        result = get_btc_price()
        self.assertIsNone(result)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_history_happy(self, mock_get, _mono, _sleep):
        raw = [[1700000000000, 40000.0, 41000.0, 39000.0, 40500.0]]
        mock_get.return_value = _ok(raw)
        from compass.crypto.coingecko import get_btc_history
        result = get_btc_history(days=1)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["time"], 1700000000000)
        self.assertEqual(row["open"], 40000.0)
        self.assertEqual(row["high"], 41000.0)
        self.assertEqual(row["low"], 39000.0)
        self.assertEqual(row["close"], 40500.0)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_eth_history_happy(self, mock_get, _mono, _sleep):
        raw = [[1700000000000, 2000.0, 2100.0, 1950.0, 2050.0]]
        mock_get.return_value = _ok(raw)
        from compass.crypto.coingecko import get_eth_history
        result = get_eth_history(days=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["close"], 2050.0)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_history_api_failure_returns_empty(self, mock_get, _mono, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.coingecko import get_btc_history
        result = get_btc_history()
        self.assertEqual(result, [])

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_dominance_happy(self, mock_get, _mono, _sleep):
        mock_get.return_value = _ok({"data": {"market_cap_percentage": {"btc": 52.3}}})
        from compass.crypto.coingecko import get_btc_dominance
        result = get_btc_dominance()
        self.assertAlmostEqual(result, 52.3)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_get_btc_dominance_api_failure_returns_none(self, mock_get, _mono, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.coingecko import get_btc_dominance
        result = get_btc_dominance()
        self.assertIsNone(result)

    @patch("compass.crypto.coingecko.time.sleep")
    @patch("compass.crypto.coingecko.time.monotonic", return_value=9999.0)
    @patch("compass.crypto.coingecko.requests.get")
    def test_retry_succeeds_on_second_attempt(self, mock_get, _mono, _sleep):
        mock_get.side_effect = [
            requests.ConnectionError("timeout"),
            _ok({"bitcoin": {"usd": 70000.0}}),
        ]
        from compass.crypto.coingecko import get_btc_price
        result = get_btc_price()
        self.assertEqual(result, 70000.0)
        self.assertEqual(mock_get.call_count, 2)


# ===========================================================================
# Fear & Greed
# ===========================================================================

class TestFearGreed(unittest.TestCase):

    @patch("compass.crypto.fear_greed.time.sleep")
    @patch("compass.crypto.fear_greed.requests.get")
    def test_get_current_happy(self, mock_get, _sleep):
        mock_get.return_value = _ok({
            "data": [{"value": "72", "value_classification": "Greed", "timestamp": "1710000000"}]
        })
        from compass.crypto.fear_greed import get_current
        result = get_current()
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 72)
        self.assertEqual(result["classification"], "Greed")
        self.assertEqual(result["timestamp"], 1710000000)

    @patch("compass.crypto.fear_greed.time.sleep")
    @patch("compass.crypto.fear_greed.requests.get")
    def test_get_current_uses_fallback_classification(self, mock_get, _sleep):
        """If value_classification is missing, _classify() is used."""
        mock_get.return_value = _ok({
            "data": [{"value": "15", "timestamp": "1710000000"}]
        })
        from compass.crypto.fear_greed import get_current
        result = get_current()
        self.assertEqual(result["classification"], "Extreme Fear")

    @patch("compass.crypto.fear_greed.time.sleep")
    @patch("compass.crypto.fear_greed.requests.get")
    def test_get_current_api_failure_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.fear_greed import get_current
        self.assertIsNone(get_current())
        self.assertEqual(mock_get.call_count, 3)

    @patch("compass.crypto.fear_greed.time.sleep")
    @patch("compass.crypto.fear_greed.requests.get")
    def test_get_history_happy(self, mock_get, _sleep):
        entries = [
            {"value": str(v), "value_classification": "Neutral", "timestamp": str(1710000000 + i)}
            for i, v in enumerate([50, 52, 48])
        ]
        mock_get.return_value = _ok({"data": entries})
        from compass.crypto.fear_greed import get_history
        result = get_history(days=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["value"], 50)

    @patch("compass.crypto.fear_greed.time.sleep")
    @patch("compass.crypto.fear_greed.requests.get")
    def test_get_history_api_failure_returns_empty(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.fear_greed import get_history
        self.assertEqual(get_history(), [])

    def test_classify_boundaries(self):
        from compass.crypto.fear_greed import _classify
        self.assertEqual(_classify(0), "Extreme Fear")
        self.assertEqual(_classify(24), "Extreme Fear")
        self.assertEqual(_classify(25), "Fear")
        self.assertEqual(_classify(44), "Fear")
        self.assertEqual(_classify(45), "Neutral")
        self.assertEqual(_classify(55), "Neutral")
        self.assertEqual(_classify(56), "Greed")
        self.assertEqual(_classify(75), "Greed")
        self.assertEqual(_classify(76), "Extreme Greed")
        self.assertEqual(_classify(100), "Extreme Greed")


# ===========================================================================
# Funding Rates (OKX)
# ===========================================================================

def _okx_ok(data_list):
    """Wrap data_list in an OKX-style response envelope."""
    return _ok({"code": "0", "msg": "", "data": data_list})


def _okx_err_envelope():
    """OKX error response (code != '0', HTTP 200)."""
    return _ok({"code": "50001", "msg": "Service unavailable", "data": []})


def _okx_current_entry(inst_id, sett_rate, funding_time_ms):
    """Build an OKX /funding-rate response entry."""
    return {
        "instId": inst_id,
        "instType": "SWAP",
        "settFundingRate": str(sett_rate),
        "fundingRate": str(sett_rate),
        "fundingTime": str(funding_time_ms),
        "nextFundingRate": "",
        "nextFundingTime": str(funding_time_ms + 28800000),
    }


def _okx_history_entry(inst_id, funding_rate, realized_rate, funding_time_ms):
    """Build an OKX /funding-rate-history response entry."""
    return {
        "instId": inst_id,
        "instType": "SWAP",
        "fundingRate": str(funding_rate),
        "realizedRate": str(realized_rate),
        "fundingTime": str(funding_time_ms),
        "method": "current_period",
    }


class TestFundingRates(unittest.TestCase):

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_btc_funding_happy(self, mock_get, _sleep):
        mock_get.return_value = _okx_ok([_okx_current_entry("BTC-USDT-SWAP", "0.00010000", 1710000000000)])
        from compass.crypto.funding_rates import get_btc_funding
        result = get_btc_funding()
        self.assertAlmostEqual(result, 0.0001)

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_eth_funding_happy(self, mock_get, _sleep):
        mock_get.return_value = _okx_ok([_okx_current_entry("ETH-USDT-SWAP", "-0.00005000", 1710000000000)])
        from compass.crypto.funding_rates import get_eth_funding
        result = get_eth_funding()
        self.assertAlmostEqual(result, -0.00005)

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_btc_funding_negative_rate(self, mock_get, _sleep):
        """Negative funding rates (shorts pay longs) parse correctly."""
        mock_get.return_value = _okx_ok([_okx_current_entry("BTC-USDT-SWAP", "-0.00075", 1710000000000)])
        from compass.crypto.funding_rates import get_btc_funding
        result = get_btc_funding()
        self.assertAlmostEqual(result, -0.00075)

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_btc_funding_http_error_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.funding_rates import get_btc_funding
        self.assertIsNone(get_btc_funding())
        self.assertEqual(mock_get.call_count, 3)

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_btc_funding_okx_error_envelope_returns_none(self, mock_get, _sleep):
        """HTTP 200 but OKX code != '0' → None."""
        mock_get.return_value = _okx_err_envelope()
        from compass.crypto.funding_rates import get_btc_funding
        self.assertIsNone(get_btc_funding())

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_btc_funding_empty_list_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _okx_ok([])
        from compass.crypto.funding_rates import get_btc_funding
        self.assertIsNone(get_btc_funding())

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_funding_history_happy(self, mock_get, _sleep):
        entries = [
            _okx_history_entry("BTC-USDT-SWAP", "0.0001", "0.0001", 1710000000000),
            _okx_history_entry("BTC-USDT-SWAP", "0.0002", "0.0002", 1710028800000),
        ]
        mock_get.return_value = _okx_ok(entries)
        from compass.crypto.funding_rates import get_funding_history
        result = get_funding_history("BTCUSDT", days=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["symbol"], "BTC-USDT-SWAP")
        self.assertAlmostEqual(result[0]["funding_rate"], 0.0001)
        self.assertEqual(result[0]["funding_time"], 1710000000000)
        self.assertAlmostEqual(result[1]["funding_rate"], 0.0002)
        self.assertEqual(result[1]["funding_time"], 1710028800000)

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_funding_history_http_error_returns_empty(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.funding_rates import get_funding_history
        self.assertEqual(get_funding_history("BTCUSDT"), [])

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_get_funding_history_okx_error_returns_empty(self, mock_get, _sleep):
        mock_get.return_value = _okx_err_envelope()
        from compass.crypto.funding_rates import get_funding_history
        self.assertEqual(get_funding_history("BTCUSDT"), [])

    @patch("compass.crypto.funding_rates.time.sleep")
    @patch("compass.crypto.funding_rates.requests.get")
    def test_retry_on_connection_error(self, mock_get, _sleep):
        mock_get.side_effect = [
            requests.ConnectionError("reset"),
            _okx_ok([_okx_current_entry("BTC-USDT-SWAP", "0.0001", 1710000000000)]),
        ]
        from compass.crypto.funding_rates import get_btc_funding
        result = get_btc_funding()
        self.assertAlmostEqual(result, 0.0001)
        self.assertEqual(mock_get.call_count, 2)

    def test_normalise_symbol_legacy_names(self):
        """Legacy Binance-style symbols map to OKX instrument IDs."""
        from compass.crypto.funding_rates import _normalise_symbol
        self.assertEqual(_normalise_symbol("BTCUSDT"), "BTC-USDT-SWAP")
        self.assertEqual(_normalise_symbol("ETHUSDT"), "ETH-USDT-SWAP")
        self.assertEqual(_normalise_symbol("BTC-USDT-SWAP"), "BTC-USDT-SWAP")

    def test_get_btc_funding_uses_okx_url(self):
        """Verify the module targets OKX, not Binance or Bybit."""
        import compass.crypto.funding_rates as fr
        self.assertIn("okx.com", fr._BASE_URL)
        self.assertNotIn("binance.com", fr._BASE_URL)
        self.assertNotIn("bybit.com", fr._BASE_URL)


class TestFundingRatesLive(unittest.TestCase):
    """Live integration tests — hit the real OKX API.

    These verify that funding rates come back non-null from US IPs
    (OKX public market data is accessible where Binance/Bybit block US).
    """

    def test_get_btc_funding_live_non_null(self):
        from compass.crypto.funding_rates import get_btc_funding
        result = get_btc_funding()
        self.assertIsNotNone(result, "BTC funding rate should be non-null from OKX")
        self.assertIsInstance(result, float)
        # Sanity: funding rate is typically in the range -0.5% to +0.5% per 8h
        self.assertGreater(result, -0.005, f"BTC funding rate {result} out of expected range")
        self.assertLess(result, 0.005, f"BTC funding rate {result} out of expected range")

    def test_get_eth_funding_live_non_null(self):
        from compass.crypto.funding_rates import get_eth_funding
        result = get_eth_funding()
        self.assertIsNotNone(result, "ETH funding rate should be non-null from OKX")
        self.assertIsInstance(result, float)
        self.assertGreater(result, -0.005, f"ETH funding rate {result} out of expected range")
        self.assertLess(result, 0.005, f"ETH funding rate {result} out of expected range")

    def test_get_btc_funding_history_live(self):
        from compass.crypto.funding_rates import get_funding_history
        result = get_funding_history("BTCUSDT", days=3)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Should return at least 1 history entry for 3 days")
        entry = result[0]
        self.assertIn("symbol", entry)
        self.assertIn("funding_rate", entry)
        self.assertIn("funding_time", entry)
        self.assertIsInstance(entry["funding_rate"], float)
        self.assertIsInstance(entry["funding_time"], int)


# ===========================================================================
# Deribit
# ===========================================================================

def _deribit_ok(result_data):
    """Wrap result_data in a Deribit-style envelope."""
    return _ok({"result": result_data})


def _make_option(expiry, strike, opt_type, oi):
    return {
        "instrument_name": f"BTC-{expiry}-{int(strike)}-{opt_type}",
        "open_interest": oi,
    }


class TestDeribit(unittest.TestCase):

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_put_call_ratio_happy(self, mock_get, _sleep):
        instruments = [
            _make_option("25MAR25", 80000, "C", 100.0),
            _make_option("25MAR25", 80000, "P", 60.0),
            _make_option("25MAR25", 90000, "C", 50.0),
            _make_option("25MAR25", 90000, "P", 40.0),
        ]
        mock_get.return_value = _deribit_ok(instruments)
        from compass.crypto.deribit import get_btc_put_call_ratio
        result = get_btc_put_call_ratio()
        # put_oi=100, call_oi=150 → ratio=100/150≈0.6667
        self.assertAlmostEqual(result, round(100.0 / 150.0, 4))

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_put_call_ratio_no_calls_returns_none(self, mock_get, _sleep):
        instruments = [_make_option("25MAR25", 80000, "P", 50.0)]
        mock_get.return_value = _deribit_ok(instruments)
        from compass.crypto.deribit import get_btc_put_call_ratio
        self.assertIsNone(get_btc_put_call_ratio())

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_put_call_ratio_api_failure_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.deribit import get_btc_put_call_ratio
        self.assertIsNone(get_btc_put_call_ratio())
        self.assertEqual(mock_get.call_count, 3)

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_max_pain_happy(self, mock_get, _sleep):
        # Three strikes: 80k, 90k, 100k
        # call_oi: 80k→200, 90k→100, 100k→50
        # put_oi:  80k→50,  90k→100, 100k→200
        instruments = [
            _make_option("25MAR25", 80000, "C", 200.0),
            _make_option("25MAR25", 90000, "C", 100.0),
            _make_option("25MAR25", 100000, "C", 50.0),
            _make_option("25MAR25", 80000, "P", 50.0),
            _make_option("25MAR25", 90000, "P", 100.0),
            _make_option("25MAR25", 100000, "P", 200.0),
        ]
        mock_get.return_value = _deribit_ok(instruments)
        from compass.crypto.deribit import get_btc_max_pain
        result = get_btc_max_pain("25MAR25")
        # Pain at 80k: no ITM calls, puts at 90k=(90k-80k)*100=1M and 100k=(100k-80k)*200=4M → 5M
        # Pain at 90k: ITM calls at 80k=(90k-80k)*200=2M; ITM puts at 100k=(100k-90k)*200=2M → 4M ← min
        # Pain at 100k: ITM calls at 80k=(100k-80k)*200=4M + 90k=(100k-90k)*100=1M = 5M; no ITM puts → 5M
        self.assertEqual(result, 90000.0)

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_max_pain_no_expiry_match_returns_none(self, mock_get, _sleep):
        instruments = [_make_option("25MAR25", 80000, "C", 100.0)]
        mock_get.return_value = _deribit_ok(instruments)
        from compass.crypto.deribit import get_btc_max_pain
        result = get_btc_max_pain("28MAR25")  # different expiry
        self.assertIsNone(result)

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_max_pain_api_failure_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.deribit import get_btc_max_pain
        self.assertIsNone(get_btc_max_pain("25MAR25"))

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_oi_by_strike_happy(self, mock_get, _sleep):
        instruments = [
            _make_option("25MAR25", 80000, "C", 150.0),
            _make_option("25MAR25", 80000, "P", 80.0),
            _make_option("25MAR25", 90000, "C", 70.0),
            _make_option("25MAR25", 90000, "P", 90.0),
            _make_option("28MAR25", 80000, "C", 999.0),  # different expiry — should be ignored
        ]
        mock_get.return_value = _deribit_ok(instruments)
        from compass.crypto.deribit import get_btc_oi_by_strike
        result = get_btc_oi_by_strike("25MAR25")
        self.assertIn(80000.0, result)
        self.assertIn(90000.0, result)
        self.assertNotIn(999.0, result)  # 28MAR25 strike excluded
        self.assertEqual(result[80000.0]["call_oi"], 150.0)
        self.assertEqual(result[80000.0]["put_oi"], 80.0)
        self.assertEqual(result[90000.0]["call_oi"], 70.0)
        self.assertEqual(result[90000.0]["put_oi"], 90.0)

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_get_btc_oi_by_strike_api_failure_returns_empty(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.deribit import get_btc_oi_by_strike
        self.assertEqual(get_btc_oi_by_strike("25MAR25"), {})

    @patch("compass.crypto.deribit.time.sleep")
    @patch("compass.crypto.deribit.requests.get")
    def test_deribit_api_error_envelope_returns_none(self, mock_get, _sleep):
        """Response without 'result' key (Deribit error envelope) → None."""
        mock_get.return_value = _ok({"error": {"code": 13009, "message": "not_open"}})
        from compass.crypto.deribit import get_btc_put_call_ratio
        self.assertIsNone(get_btc_put_call_ratio())

    def test_parse_instrument_valid(self):
        from compass.crypto.deribit import _parse_instrument
        expiry, strike, opt_type = _parse_instrument("BTC-25MAR25-100000-C")
        self.assertEqual(expiry, "25MAR25")
        self.assertEqual(strike, 100000.0)
        self.assertEqual(opt_type, "C")

    def test_parse_instrument_invalid(self):
        from compass.crypto.deribit import _parse_instrument
        self.assertEqual(_parse_instrument("BTC-PERP"), (None, None, None))
        self.assertEqual(_parse_instrument("BTC-25MAR25-BADSTRIKE-C"), (None, None, None))
        self.assertEqual(_parse_instrument("BTC-25MAR25-100000-X"), (None, None, None))


# ===========================================================================
# DeFiLlama
# ===========================================================================

def _llama_asset_list(assets):
    return _ok({"peggedAssets": assets})


def _llama_asset(symbol, asset_id, supply):
    return {
        "id": asset_id,
        "symbol": symbol,
        "circulating": {"peggedUSD": supply},
    }


def _llama_history_resp(entries):
    """entries = list of (unix_ts, supply)"""
    return _ok({
        "tokens": [
            {"date": ts, "circulating": {"peggedUSD": supply}}
            for ts, supply in entries
        ]
    })


class TestDefiLlama(unittest.TestCase):

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_total_stablecoin_supply_happy(self, mock_get, _sleep):
        assets = [
            _llama_asset("USDT", "1", 90_000_000_000.0),
            _llama_asset("USDC", "2", 35_000_000_000.0),
            _llama_asset("DAI",  "3",  5_000_000_000.0),
            _llama_asset("BUSD", "4",  2_000_000_000.0),  # not tracked — ignored
        ]
        mock_get.return_value = _llama_asset_list(assets)
        from compass.crypto.defi_llama import get_total_stablecoin_supply
        result = get_total_stablecoin_supply()
        self.assertAlmostEqual(result, 130_000_000_000.0)

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_total_supply_only_usdt_present(self, mock_get, _sleep):
        """If only USDT is present, return its supply (not None)."""
        assets = [_llama_asset("USDT", "1", 90_000_000_000.0)]
        mock_get.return_value = _llama_asset_list(assets)
        from compass.crypto.defi_llama import get_total_stablecoin_supply
        result = get_total_stablecoin_supply()
        self.assertAlmostEqual(result, 90_000_000_000.0)

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_total_supply_no_tracked_assets_returns_none(self, mock_get, _sleep):
        assets = [_llama_asset("BUSD", "4", 1_000_000_000.0)]
        mock_get.return_value = _llama_asset_list(assets)
        from compass.crypto.defi_llama import get_total_stablecoin_supply
        self.assertIsNone(get_total_stablecoin_supply())

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_total_supply_api_failure_returns_none(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.defi_llama import get_total_stablecoin_supply
        self.assertIsNone(get_total_stablecoin_supply())
        self.assertEqual(mock_get.call_count, 3)

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_stablecoin_history_happy(self, mock_get, _sleep):
        # First call = asset list; next 3 calls = per-stablecoin history
        import time as _time
        now_ts = int(_time.time())
        day = 86400
        assets = [
            _llama_asset("USDT", "1", 90e9),
            _llama_asset("USDC", "2", 35e9),
            _llama_asset("DAI",  "3",  5e9),
        ]
        usdt_hist = [(now_ts - day, 89e9), (now_ts, 90e9)]
        usdc_hist = [(now_ts - day, 34e9), (now_ts, 35e9)]
        dai_hist  = [(now_ts - day,  4e9), (now_ts,  5e9)]
        mock_get.side_effect = [
            _llama_asset_list(assets),
            _llama_history_resp(usdt_hist),
            _llama_history_resp(usdc_hist),
            _llama_history_resp(dai_hist),
        ]
        from compass.crypto.defi_llama import get_stablecoin_history
        result = get_stablecoin_history(days=90)
        # Should have 2 dates, each summing all 3 stablecoins
        self.assertEqual(len(result), 2)
        # Earlier date
        self.assertAlmostEqual(result[0]["total_supply"], 89e9 + 34e9 + 4e9)
        # Later date
        self.assertAlmostEqual(result[1]["total_supply"], 90e9 + 35e9 + 5e9)
        # Sorted ascending
        self.assertLess(result[0]["date"], result[1]["date"])

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_stablecoin_history_asset_list_failure_returns_empty(self, mock_get, _sleep):
        mock_get.return_value = _err()
        from compass.crypto.defi_llama import get_stablecoin_history
        self.assertEqual(get_stablecoin_history(), [])

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_stablecoin_history_partial_history_failure(self, mock_get, _sleep):
        """If one asset's history fetch fails, others still contribute."""
        import time as _time
        now_ts = int(_time.time())
        assets = [
            _llama_asset("USDT", "1", 90e9),
            _llama_asset("USDC", "2", 35e9),
        ]
        mock_get.side_effect = [
            _llama_asset_list(assets),
            _llama_history_resp([(now_ts, 90e9)]),  # USDT ok
            _err(),                                  # USDC fails (3 retries)
            _err(),
            _err(),
        ]
        from compass.crypto.defi_llama import get_stablecoin_history
        result = get_stablecoin_history(days=90)
        # USDT history still returned
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["total_supply"], 90e9)

    @patch("compass.crypto.defi_llama.time.sleep")
    @patch("compass.crypto.defi_llama.requests.get")
    def test_get_stablecoin_history_filters_old_dates(self, mock_get, _sleep):
        """Entries older than `days` are excluded."""
        import time as _time
        now_ts = int(_time.time())
        old_ts = now_ts - 200 * 86400  # 200 days ago
        assets = [_llama_asset("USDT", "1", 90e9)]
        mock_get.side_effect = [
            _llama_asset_list(assets),
            _llama_history_resp([(old_ts, 80e9), (now_ts, 90e9)]),
        ]
        from compass.crypto.defi_llama import get_stablecoin_history
        result = get_stablecoin_history(days=90)
        # old_ts is 200 days ago — outside the 90-day window
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["total_supply"], 90e9)


if __name__ == "__main__":
    unittest.main()
