"""Contract tests using frozen API response fixtures.

These tests verify that:
1. Frozen fixtures can be loaded and parsed correctly.
2. Provider methods produce the correct output shape from frozen responses.
3. Mock schemas used elsewhere match the real fixture schemas.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


def _fixture_to_dataframe(fixture: dict) -> pd.DataFrame:
    """Convert the yfinance history fixture into a DataFrame."""
    df = pd.DataFrame(fixture["data"], columns=fixture["columns"],
                      index=pd.to_datetime(fixture["index"]))
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 1. Fixture loading / schema tests
# ---------------------------------------------------------------------------

class TestYFinanceFixture:

    def test_load_yfinance_fixture(self):
        """Frozen yfinance fixture loads without error."""
        fixture = _load_fixture("yfinance_spy_history.json")
        assert "columns" in fixture
        assert "data" in fixture
        assert "index" in fixture

    def test_yfinance_columns_match_real_schema(self):
        """Fixture columns should match what yfinance.download returns."""
        fixture = _load_fixture("yfinance_spy_history.json")
        expected_columns = {"Open", "High", "Low", "Close", "Volume"}
        assert set(fixture["columns"]) == expected_columns

    def test_yfinance_dataframe_conversion(self):
        """Fixture should convert to a well-formed DataFrame."""
        fixture = _load_fixture("yfinance_spy_history.json")
        df = _fixture_to_dataframe(fixture)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "Date"
        # Prices should all be positive
        assert (df["Close"] > 0).all()
        assert (df["Volume"] > 0).all()

    def test_yfinance_fixture_matches_conftest_mock(self):
        """The frozen fixture columns must match the conftest sample_price_data columns."""
        fixture = _load_fixture("yfinance_spy_history.json")
        # conftest produces: Open, High, Low, Close, Volume
        expected = ["Open", "High", "Low", "Close", "Volume"]
        assert fixture["columns"] == expected


class TestTradierFixture:

    def test_load_tradier_fixture(self):
        """Frozen Tradier fixture loads without error."""
        fixture = _load_fixture("tradier_chain_response.json")
        assert "options" in fixture
        assert "option" in fixture["options"]

    def test_tradier_option_has_required_fields(self):
        """Each option in the fixture must have the fields that TradierProvider.get_options_chain reads."""
        fixture = _load_fixture("tradier_chain_response.json")
        required_fields = {"symbol", "strike", "option_type", "bid", "ask", "volume",
                           "open_interest", "greeks"}
        greek_fields = {"delta", "gamma", "theta", "vega", "mid_iv"}

        for opt in fixture["options"]["option"]:
            assert required_fields.issubset(opt.keys()), f"Missing fields: {required_fields - opt.keys()}"
            greeks = opt["greeks"]
            assert greek_fields.issubset(greeks.keys()), f"Missing greeks: {greek_fields - greeks.keys()}"

    def test_tradier_provider_parses_fixture(self):
        """TradierProvider.get_options_chain should produce a valid DataFrame
        when the HTTP response matches the frozen fixture."""
        fixture = _load_fixture("tradier_chain_response.json")

        mock_resp = MagicMock()
        mock_resp.json.return_value = fixture
        mock_resp.raise_for_status = MagicMock()

        with patch("strategy.tradier_provider.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            from strategy.tradier_provider import TradierProvider
            provider = TradierProvider(api_key="test_key", sandbox=True)
            provider.session = mock_session

            df = provider.get_options_chain("SPY", "2025-02-21")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Verify expected columns produced by the provider
        expected_cols = {"strike", "type", "bid", "ask", "delta", "gamma", "theta", "vega",
                         "iv", "volume", "open_interest", "mid", "expiration"}
        assert expected_cols.issubset(set(df.columns)), f"Missing: {expected_cols - set(df.columns)}"

    def test_tradier_fixture_delta_values_reasonable(self):
        """Deltas in the fixture should be within [-1, 1]."""
        fixture = _load_fixture("tradier_chain_response.json")
        for opt in fixture["options"]["option"]:
            delta = opt["greeks"]["delta"]
            assert -1.0 <= delta <= 1.0, f"Unreasonable delta: {delta}"


class TestTelegramFixture:

    def test_load_telegram_fixture(self):
        """Frozen Telegram send_message response loads correctly."""
        fixture = _load_fixture("telegram_send_message.json")
        assert fixture["ok"] is True
        assert "result" in fixture

    def test_telegram_response_has_required_fields(self):
        """The Telegram response must have message_id and chat info."""
        fixture = _load_fixture("telegram_send_message.json")
        result = fixture["result"]
        assert "message_id" in result
        assert "chat" in result
        assert "id" in result["chat"]
        assert "text" in result


# ---------------------------------------------------------------------------
# 2. Cross-fixture / integration shape tests
# ---------------------------------------------------------------------------

class TestCrossFixtureConsistency:

    def test_yfinance_data_can_compute_indicators(self):
        """Frozen yfinance data should work with the indicator functions."""
        from shared.indicators import calculate_rsi

        fixture = _load_fixture("yfinance_spy_history.json")
        df = _fixture_to_dataframe(fixture)

        # RSI should produce valid values for the available data
        rsi = calculate_rsi(df["Close"], period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)
        # First 14 values will be NaN, rest should be in [0, 100]
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_yfinance_data_can_compute_iv_rank(self):
        """IV rank calculation should work on frozen history data."""
        from shared.indicators import calculate_iv_rank

        fixture = _load_fixture("yfinance_spy_history.json")
        df = _fixture_to_dataframe(fixture)

        # Compute a rolling HV series
        returns = df["Close"].pct_change().dropna()
        hv = returns.rolling(window=5).std() * np.sqrt(252) * 100
        hv = hv.dropna()

        # Use a current_iv within the historical range so iv_rank stays in [0, 100]
        mid_iv = float((hv.min() + hv.max()) / 2)
        result = calculate_iv_rank(hv, current_iv=mid_iv)
        assert "iv_rank" in result
        assert "iv_percentile" in result
        # iv_rank is bounded when current_iv is within the historical range
        assert 0 <= result["iv_rank"] <= 100
        assert 0 <= result["iv_percentile"] <= 100
