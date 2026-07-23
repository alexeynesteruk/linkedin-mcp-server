import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.debug_trace import (
    _safe_source_profile_dir,
    cleanup_trace_dir,
    garbage_collect_trace_runs,
    get_trace_dir,
    mark_trace_for_retention,
    record_page_trace,
    reset_trace_state_for_testing,
)


def setup_function():
    reset_trace_state_for_testing()


def teardown_function():
    reset_trace_state_for_testing()


def test_get_trace_dir_creates_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    trace_dir = get_trace_dir()

    assert trace_dir is not None
    assert trace_dir.exists()
    assert "trace-runs" in str(trace_dir)


def test_cleanup_trace_dir_removes_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = get_trace_dir()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert not trace_dir.exists()


def test_mark_trace_for_retention_keeps_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = mark_trace_for_retention()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_explicit_trace_dir_is_preserved(monkeypatch, tmp_path):
    trace_dir = tmp_path / "explicit-trace"
    monkeypatch.setenv("LINKEDIN_DEBUG_TRACE_DIR", str(trace_dir))

    resolved = get_trace_dir()
    assert resolved == trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_trace_mode_off_disables_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_MODE", "off")

    assert get_trace_dir() is None


@pytest.mark.asyncio
async def test_reset_trace_state_resets_step_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="LinkedIn")
    page.evaluate = AsyncMock(return_value="Feed")
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=locator)
    page.context.cookies = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()

    await record_page_trace(page, "first")
    trace_dir = get_trace_dir()
    assert trace_dir is not None
    first_payload = json.loads((trace_dir / "trace.jsonl").read_text().splitlines()[0])
    assert first_payload["step_id"] == 1

    reset_trace_state_for_testing()
    monkeypatch.setenv("USER_DATA_DIR", str((tmp_path / "second") / "profile"))

    await record_page_trace(page, "first-again")
    second_trace_dir = get_trace_dir()
    assert second_trace_dir is not None
    second_payload = json.loads(
        (second_trace_dir / "trace.jsonl").read_text().splitlines()[0]
    )
    assert second_payload["step_id"] == 1


def test_safe_source_profile_dir_ignores_generic_env_fallback(monkeypatch):
    monkeypatch.setenv("USER_DATA_DIR", "/tmp/unrelated-user-data")
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace.get_source_profile_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _safe_source_profile_dir() == Path("~/.linkedin-mcp/profile").expanduser()


def test_garbage_collect_deletes_old_empty_runs(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("USER_DATA_DIR", str(profile))
    reset_trace_state_for_testing()

    root = tmp_path / "trace-runs"
    root.mkdir()
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace._trace_root",
        lambda: root,
    )

    old_empty = root / "run-old-empty"
    old_empty.mkdir()
    (old_empty / "server.log").write_text("")
    # Age the directory
    old_ts = 1_000_000.0
    import os

    os.utime(old_empty, (old_ts, old_ts))
    os.utime(old_empty / "server.log", (old_ts, old_ts))

    fresh_empty = root / "run-fresh-empty"
    fresh_empty.mkdir()
    (fresh_empty / "server.log").write_text("")

    kept = root / "run-with-content"
    kept.mkdir()
    (kept / "server.log").write_text("x" * 200)
    os.utime(kept, (old_ts, old_ts))

    stats = garbage_collect_trace_runs(
        max_age_days=1.0,
        max_runs=200,
        now=old_ts + 10 * 86400,
    )

    assert stats["deleted_empty"] == 1
    assert not old_empty.exists()
    assert fresh_empty.exists()
    assert kept.exists()


def test_garbage_collect_respects_max_runs_preferring_empty(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("USER_DATA_DIR", str(profile))
    reset_trace_state_for_testing()

    root = tmp_path / "trace-runs"
    root.mkdir()
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace._trace_root",
        lambda: root,
    )

    import os
    import time

    now = time.time()
    for i in range(5):
        d = root / f"run-e{i}"
        d.mkdir()
        (d / "server.log").write_text("")
        os.utime(d, (now - (10 - i), now - (10 - i)))

    content = root / "run-content"
    content.mkdir()
    (content / "server.log").write_text("payload" * 20)

    stats = garbage_collect_trace_runs(max_age_days=0, max_runs=3, now=now)

    assert stats["deleted_over_cap"] >= 2
    assert content.exists()
    remaining = list(root.iterdir())
    assert len(remaining) <= 3


def test_garbage_collect_skips_active_trace_dir(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("USER_DATA_DIR", str(profile))
    reset_trace_state_for_testing()

    root = tmp_path / "trace-runs"
    root.mkdir()
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace._trace_root",
        lambda: root,
    )

    active = get_trace_dir()
    assert active is not None
    # Force active under our root for the test
    import linkedin_mcp_server.debug_trace as dt

    protected = root / "run-active"
    protected.mkdir()
    (protected / "server.log").write_text("")
    dt._TRACE_DIR = protected

    old = root / "run-old"
    old.mkdir()
    (old / "server.log").write_text("")
    import os

    os.utime(old, (1_000_000.0, 1_000_000.0))

    garbage_collect_trace_runs(
        max_age_days=1.0, max_runs=200, now=1_000_000.0 + 20 * 86400
    )

    assert protected.exists()
    assert not old.exists()
