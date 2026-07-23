"""Tests for Chromium SingletonLock detection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from linkedin_mcp_server.drivers.browser import (
    profile_singleton_lock_held,
    warn_if_profile_lock_held,
)


class TestProfileSingletonLockHeld:
    def test_missing_lock(self, tmp_path: Path):
        assert profile_singleton_lock_held(tmp_path) is False

    def test_live_pid_symlink(self, tmp_path: Path):
        lock = tmp_path / "SingletonLock"
        lock.symlink_to(f"testhost-{os.getpid()}")
        assert profile_singleton_lock_held(tmp_path) is True

    def test_dead_pid_symlink(self, tmp_path: Path):
        lock = tmp_path / "SingletonLock"
        # PID 1 is init/launchd and usually exists; use a high unused pid.
        dead_pid = 2_147_000_000
        lock.symlink_to(f"testhost-{dead_pid}")
        with patch("os.kill", side_effect=ProcessLookupError):
            assert profile_singleton_lock_held(tmp_path) is False

    def test_plain_file_lock_counts_as_held(self, tmp_path: Path):
        lock = tmp_path / "SingletonLock"
        lock.write_text("opaque")
        assert profile_singleton_lock_held(tmp_path) is True


class TestWarnIfProfileLockHeld:
    def test_warns_when_held(self, tmp_path: Path, caplog):
        lock = tmp_path / "SingletonLock"
        lock.symlink_to(f"testhost-{os.getpid()}")
        import logging

        with caplog.at_level(logging.WARNING):
            assert warn_if_profile_lock_held(tmp_path) is True
        assert any("SingletonLock is held" in r.message for r in caplog.records)

    def test_silent_when_clear(self, tmp_path: Path, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert warn_if_profile_lock_held(tmp_path) is False
        assert not any("SingletonLock" in r.message for r in caplog.records)
