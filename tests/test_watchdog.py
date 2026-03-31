"""Tests for scripts/watchdog.py — INF-3 auto-restart watchdog."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import yaml

# Ensure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import watchdog as wd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ET = timezone(timedelta(hours=-4))


@pytest.fixture
def sample_experiments_yaml(tmp_path):
    """Create a minimal experiments.yaml in a temp dir."""
    data = {
        "experiments": {
            "exp400": {
                "description": "Champion",
                "status": "active",
                "env_file": ".env.champion",
                "config_file": "configs/paper_champion.yaml",
                "tmux_session": "exp400",
                "db_path": "data/pilotai_champion.db",
            },
            "exp036": {
                "description": "Old experiment",
                "status": "stopped",
                "tmux_session": None,
                "db_path": "data/pilotai_exp036.db",
            },
        }
    }
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text(yaml.dump(data))
    return str(config_path)


@pytest.fixture
def heartbeat_dir(tmp_path):
    """Create a data dir with a heartbeat file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return str(data_dir)


# ---------------------------------------------------------------------------
# Tests: tmux
# ---------------------------------------------------------------------------


class TestTmuxSessionAlive:
    @mock.patch("subprocess.run")
    def test_alive(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0)
        assert wd.tmux_session_alive("exp400") is True
        mock_run.assert_called_once()

    @mock.patch("subprocess.run")
    def test_dead(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1)
        assert wd.tmux_session_alive("exp400") is False

    def test_empty_session_name(self):
        assert wd.tmux_session_alive("") is False
        assert wd.tmux_session_alive(None) is False

    @mock.patch("subprocess.run", side_effect=Exception("tmux not found"))
    def test_exception(self, mock_run):
        assert wd.tmux_session_alive("exp400") is False


# ---------------------------------------------------------------------------
# Tests: heartbeat
# ---------------------------------------------------------------------------


class TestCheckHeartbeat:
    def test_valid_heartbeat(self, heartbeat_dir):
        ts = datetime.now(timezone.utc).isoformat()
        Path(heartbeat_dir, ".last_scan_exp400").write_text(ts)
        result = wd.check_heartbeat(heartbeat_dir, "exp400")
        assert result is not None

    def test_missing_heartbeat(self, heartbeat_dir):
        result = wd.check_heartbeat(heartbeat_dir, "nonexistent")
        assert result is None

    def test_corrupt_heartbeat(self, heartbeat_dir):
        Path(heartbeat_dir, ".last_scan_bad").write_text("not-a-date")
        result = wd.check_heartbeat(heartbeat_dir, "bad")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: market hours
# ---------------------------------------------------------------------------


class TestMarketHours:
    def test_weekday_during_hours(self):
        # Wednesday 10:00 AM ET
        dt = datetime(2026, 3, 18, 10, 0, tzinfo=ET)
        assert wd.is_market_hours(dt) is True

    def test_weekday_before_hours(self):
        dt = datetime(2026, 3, 18, 8, 0, tzinfo=ET)
        assert wd.is_market_hours(dt) is False

    def test_weekend(self):
        # Saturday
        dt = datetime(2026, 3, 14, 10, 0, tzinfo=ET)
        assert wd.is_market_hours(dt) is False

    def test_after_close(self):
        dt = datetime(2026, 3, 18, 16, 0, tzinfo=ET)
        assert wd.is_market_hours(dt) is False


# ---------------------------------------------------------------------------
# Tests: Alpaca API
# ---------------------------------------------------------------------------


class TestAlpacaApi:
    @mock.patch("urllib.request.urlopen")
    def test_api_ok(self, mock_urlopen, tmp_path):
        env_file = tmp_path / ".env.test"
        env_file.write_text("APCA_API_KEY_ID=testkey\nAPCA_API_SECRET_KEY=testsecret\n")
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert wd.check_alpaca_api(str(env_file)) is True

    @mock.patch("urllib.request.urlopen", side_effect=Exception("timeout"))
    def test_api_down(self, mock_urlopen, tmp_path):
        env_file = tmp_path / ".env.test"
        env_file.write_text("APCA_API_KEY_ID=testkey\nAPCA_API_SECRET_KEY=testsecret\n")
        assert wd.check_alpaca_api(str(env_file)) is False

    def test_missing_keys(self, tmp_path):
        env_file = tmp_path / ".env.empty"
        env_file.write_text("FOO=bar\n")
        assert wd.check_alpaca_api(str(env_file)) is False


# ---------------------------------------------------------------------------
# Tests: restart
# ---------------------------------------------------------------------------


class TestRestartTmuxSession:
    @mock.patch("subprocess.run")
    def test_restart_success(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0)
        result = wd.restart_tmux_session("exp400", "/tmp/proj", ".env.x", "cfg.yaml", "data/db.db")
        assert result is True

    @mock.patch("subprocess.run")
    def test_restart_failure(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1)
        result = wd.restart_tmux_session("exp400", "/tmp/proj", ".env.x", "cfg.yaml", "data/db.db")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: full watchdog run
# ---------------------------------------------------------------------------


class TestRunWatchdog:
    @mock.patch.object(wd, "send_telegram_alert", return_value=True)
    @mock.patch.object(wd, "check_alpaca_api", return_value=True)
    @mock.patch.object(wd, "restart_tmux_session", return_value=True)
    @mock.patch.object(wd, "tmux_session_alive", return_value=True)
    def test_all_healthy(self, mock_tmux, mock_restart, mock_alpaca, mock_tg, sample_experiments_yaml):
        results = wd.run_watchdog(sample_experiments_yaml)
        assert "experiments" in results
        assert results["experiments"]["exp400"]["tmux_alive"] is True
        assert results["experiments"]["exp036"]["monitored"] is False
        assert len(results["restarts"]) == 0

    @mock.patch.object(wd, "send_telegram_alert", return_value=True)
    @mock.patch.object(wd, "check_alpaca_api", return_value=True)
    @mock.patch.object(wd, "restart_tmux_session", return_value=True)
    @mock.patch.object(wd, "tmux_session_alive", return_value=False)
    def test_dead_session_restarted(self, mock_tmux, mock_restart, mock_alpaca, mock_tg, sample_experiments_yaml):
        results = wd.run_watchdog(sample_experiments_yaml)
        assert results["experiments"]["exp400"]["restarted"] is True
        assert "exp400" in results["restarts"]
        assert mock_tg.called

    @mock.patch.object(wd, "send_telegram_alert", return_value=True)
    @mock.patch.object(wd, "check_alpaca_api", return_value=False)
    @mock.patch.object(wd, "tmux_session_alive", return_value=True)
    def test_alpaca_down_alert(self, mock_tmux, mock_alpaca, mock_tg, sample_experiments_yaml):
        results = wd.run_watchdog(sample_experiments_yaml)
        assert results["experiments"]["exp400"]["alpaca_api_ok"] is False
        assert any("Alpaca" in a for a in results["alerts"])


# ---------------------------------------------------------------------------
# Tests: load experiments
# ---------------------------------------------------------------------------


class TestLoadExperiments:
    def test_load(self, sample_experiments_yaml):
        exps = wd.load_experiments(sample_experiments_yaml)
        assert "exp400" in exps
        assert exps["exp400"]["status"] == "active"


# ---------------------------------------------------------------------------
# Tests: env file parser
# ---------------------------------------------------------------------------


class TestParseEnvFile:
    def test_basic(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text('KEY1=val1\nKEY2="val2"\n# comment\n\nKEY3=\'val3\'\n')
        result = wd._parse_env_file(str(f))
        assert result["KEY1"] == "val1"
        assert result["KEY2"] == "val2"
        assert result["KEY3"] == "val3"

    def test_missing_file(self):
        result = wd._parse_env_file("/nonexistent/.env")
        assert result == {}
