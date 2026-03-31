"""Tests for configuration loading and validation."""
import os
from unittest.mock import patch

import pytest
import yaml

from utils import load_config, validate_config


class TestLoadConfig:

    def test_load_config_success(self, tmp_path):
        """load_config should return a dict from a valid YAML file."""
        cfg = {
            'tickers': ['SPY'],
            'strategy': {'min_dte': 30, 'max_dte': 45},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg))

        with patch('dotenv.load_dotenv'):
            result = load_config(str(cfg_file))

        assert result['tickers'] == ['SPY']
        assert result['strategy']['min_dte'] == 30

    def test_load_config_missing_file(self):
        """load_config should raise FileNotFoundError for a missing path."""
        with patch('dotenv.load_dotenv'):
            with pytest.raises(FileNotFoundError):
                load_config('/nonexistent/path/config.yaml')


class TestValidateConfig:

    def test_validate_config_valid(self, sample_config):
        """A complete, well-formed config should pass validation (no exception)."""
        validate_config(sample_config)  # Should not raise

    def test_validate_config_missing_section(self, sample_config):
        """Removing a required section should raise ValueError."""
        del sample_config['strategy']
        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config(sample_config)

    def test_validate_config_bad_dte(self, sample_config):
        """min_dte >= max_dte should raise ValueError."""
        sample_config['strategy']['min_dte'] = 50
        sample_config['strategy']['max_dte'] = 45
        with pytest.raises(ValueError, match="min_dte must be less than max_dte"):
            validate_config(sample_config)

    def test_validate_config_bad_delta(self, sample_config):
        """min_delta >= max_delta should raise ValueError."""
        sample_config['strategy']['min_delta'] = 0.20
        sample_config['strategy']['max_delta'] = 0.15
        with pytest.raises(ValueError, match="min_delta must be less than max_delta"):
            validate_config(sample_config)

    def test_validate_config_bad_account(self, sample_config):
        """account_size <= 0 should raise ValueError."""
        sample_config['risk']['account_size'] = -500
        with pytest.raises(ValueError, match="account_size must be positive"):
            validate_config(sample_config)

    def test_env_var_resolution(self, tmp_path):
        """${ENV_VAR} references in config strings should be resolved."""
        cfg = {
            'tickers': ['SPY'],
            'api_key': '${MY_TEST_API_KEY}',
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg))

        with patch('dotenv.load_dotenv'):
            with patch.dict(os.environ, {'MY_TEST_API_KEY': 'secret123'}):
                result = load_config(str(cfg_file))

        assert result['api_key'] == 'secret123'
