"""Tests for the Kelly-criterion position sizer."""

from compass.sizing import PositionSizer


class TestPositionSizer:
    """Tests for PositionSizer.calculate_position_size."""

    def _make_sizer(self, **kwargs):
        defaults = dict(
            max_position_size=0.10,
            kelly_fraction=0.25,
            max_portfolio_risk=0.20,
            max_correlated_exposure=0.15,
        )
        defaults.update(kwargs)
        return PositionSizer(**defaults)

    # ----- core kelly tests -----

    def test_kelly_positive_ev(self):
        """A positive-EV trade should produce a non-zero position size."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            win_probability=0.80,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        assert result['recommended_size'] > 0

    def test_kelly_negative_ev_returns_zero(self):
        """A negative-EV trade (low win prob, bad odds) should return size 0."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            win_probability=0.20,
            expected_return=0.10,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        assert result['recommended_size'] == 0.0

    def test_kelly_fraction_applied(self):
        """Fractional Kelly should reduce the position relative to full Kelly."""
        sizer_full = self._make_sizer(kelly_fraction=1.0, max_position_size=1.0)
        sizer_quarter = self._make_sizer(kelly_fraction=0.25, max_position_size=1.0)

        full = sizer_full.calculate_position_size(
            win_probability=0.70,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        quarter = sizer_quarter.calculate_position_size(
            win_probability=0.70,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        assert quarter['recommended_size'] < full['recommended_size']

    def test_max_position_cap(self):
        """Position size should never exceed max_position_size."""
        sizer = self._make_sizer(max_position_size=0.05, kelly_fraction=1.0)
        result = sizer.calculate_position_size(
            win_probability=0.90,
            expected_return=2.0,
            expected_loss=-0.5,
            ml_confidence=1.0,
        )
        assert result['recommended_size'] <= 0.05

    def test_confidence_scaling(self):
        """Lower ML confidence should reduce the recommended size."""
        sizer = self._make_sizer()
        high_conf = sizer.calculate_position_size(
            win_probability=0.80,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        low_conf = sizer.calculate_position_size(
            win_probability=0.80,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=0.50,
        )
        assert low_conf['recommended_size'] <= high_conf['recommended_size']

    def test_zero_probability(self):
        """Zero win probability should return zero size."""
        sizer = self._make_sizer()
        result = sizer.calculate_position_size(
            win_probability=0.0,
            expected_return=0.30,
            expected_loss=-1.0,
            ml_confidence=1.0,
        )
        assert result['recommended_size'] == 0.0

    def test_portfolio_constraint(self):
        """Adding a position when portfolio is near capacity should limit size."""
        sizer = self._make_sizer(max_portfolio_risk=0.10)

        existing = [
            {'ticker': 'SPY', 'position_size': 0.08},
        ]
        result = sizer.calculate_position_size(
            win_probability=0.80,
            expected_return=0.50,
            expected_loss=-1.0,
            ml_confidence=1.0,
            current_positions=existing,
            ticker='QQQ',
        )
        # Available risk is only 0.02, so size should be at most 0.02
        assert result['recommended_size'] <= 0.02 + 1e-6
